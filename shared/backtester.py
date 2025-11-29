#!/usr/bin/env python3
"""
Backtesting Engine
Run historical simulations for all trading tiers.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Type
from dataclasses import dataclass, field
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade in backtest."""
    trade_id: int
    symbol: str
    action: str  # LONG or SHORT
    entry_time: datetime
    entry_price: float
    stop_loss: float
    take_profit: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # TP_HIT, SL_HIT, EXPIRED, CLOSED
    pnl_percent: Optional[float] = None
    pnl_absolute: Optional[float] = None
    duration_minutes: Optional[int] = None
    score: float = 0.0
    indicators: Dict = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    strategy_name: str
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    expired_trades: int

    # Performance metrics
    win_rate: float
    total_pnl_percent: float
    total_pnl_absolute: float
    average_win_percent: float
    average_loss_percent: float
    largest_win_percent: float
    largest_loss_percent: float

    # Risk metrics
    max_drawdown_percent: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    profit_factor: float
    sharpe_ratio: float

    # Time metrics
    average_trade_duration_minutes: float

    # All trades
    trades: List[Trade] = field(default_factory=list)

    # Equity curve
    equity_curve: List[Tuple[datetime, float]] = field(default_factory=list)


class BacktestEngine:
    """
    Core backtesting engine that simulates trading strategies
    against historical data.
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.initial_capital = self.config.get('initial_capital', 10000)
        self.position_size_percent = self.config.get('position_size_percent', 2.0)
        self.max_open_trades = self.config.get('max_open_trades', 5)
        self.commission_percent = self.config.get('commission_percent', 0.1)
        self.slippage_percent = self.config.get('slippage_percent', 0.05)

        self.trades: List[Trade] = []
        self.open_trades: List[Trade] = []
        self.trade_counter = 0
        self.capital = self.initial_capital
        self.equity_curve: List[Tuple[datetime, float]] = []

    def reset(self):
        """Reset backtest state."""
        self.trades = []
        self.open_trades = []
        self.trade_counter = 0
        self.capital = self.initial_capital
        self.equity_curve = []

    def run(self,
            df: pd.DataFrame,
            indicator_calculator,
            strategy,
            symbol: str) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            df: DataFrame with OHLCV data
            indicator_calculator: Tier-specific indicator calculator instance
            strategy: Tier-specific strategy instance
            symbol: Trading pair symbol

        Returns:
            BacktestResult with all metrics
        """
        self.reset()

        if df is None or len(df) < 100:
            raise ValueError("Insufficient data for backtesting (need 100+ candles)")

        # Ensure we have datetime index
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')

        # Calculate all indicators on full dataset
        df_with_indicators = indicator_calculator.calculate_all(df.copy())

        if df_with_indicators is None:
            raise ValueError("Failed to calculate indicators")

        # Iterate through each candle (skip warmup period)
        warmup = 50

        for i in range(warmup, len(df_with_indicators)):
            current_time = df_with_indicators.index[i]
            current_candle = df_with_indicators.iloc[i]

            # Get data up to current point (no look-ahead bias)
            historical_df = df_with_indicators.iloc[:i+1]

            # Check open trades for TP/SL
            self._check_open_trades(current_candle, current_time)

            # Check for new signals (if we have capacity)
            if len(self.open_trades) < self.max_open_trades:
                signal = strategy.check_signals(symbol, historical_df)

                if signal:
                    self._open_trade(signal, current_candle, current_time)

            # Record equity
            equity = self._calculate_equity(current_candle)
            self.equity_curve.append((current_time, equity))

        # Close any remaining open trades at end
        if len(self.open_trades) > 0:
            final_candle = df_with_indicators.iloc[-1]
            final_time = df_with_indicators.index[-1]
            for trade in self.open_trades[:]:
                self._close_trade(trade, final_candle['close'], final_time, 'END_OF_DATA')

        # Calculate results
        return self._calculate_results(
            strategy_name=strategy.strategy_name,
            symbol=symbol,
            timeframe=self.config.get('timeframe', '15m'),
            start_date=df_with_indicators.index[warmup],
            end_date=df_with_indicators.index[-1]
        )

    def _open_trade(self, signal: Dict, candle: pd.Series, time: datetime):
        """Open a new trade based on signal."""
        self.trade_counter += 1

        # Apply slippage to entry
        entry_price = signal['entry_price']
        if signal['action'] == 'LONG':
            entry_price *= (1 + self.slippage_percent / 100)
        else:
            entry_price *= (1 - self.slippage_percent / 100)

        trade = Trade(
            trade_id=self.trade_counter,
            symbol=signal['symbol'],
            action=signal['action'],
            entry_time=time,
            entry_price=entry_price,
            stop_loss=signal['stop_loss'],
            take_profit=signal['take_profit'],
            score=signal.get('score', 0),
            indicators=signal.get('indicators', {})
        )

        self.open_trades.append(trade)
        logger.debug(f"Opened {trade.action} trade #{trade.trade_id} at {entry_price}")

    def _check_open_trades(self, candle: pd.Series, time: datetime):
        """Check open trades for TP/SL hits."""
        high = candle['high']
        low = candle['low']

        for trade in self.open_trades[:]:  # Copy list to allow modification
            if trade.action == 'LONG':
                # Check stop loss first (worst case)
                if low <= trade.stop_loss:
                    self._close_trade(trade, trade.stop_loss, time, 'SL_HIT')
                # Check take profit
                elif high >= trade.take_profit:
                    self._close_trade(trade, trade.take_profit, time, 'TP_HIT')
            else:  # SHORT
                # Check stop loss first
                if high >= trade.stop_loss:
                    self._close_trade(trade, trade.stop_loss, time, 'SL_HIT')
                # Check take profit
                elif low <= trade.take_profit:
                    self._close_trade(trade, trade.take_profit, time, 'TP_HIT')

    def _close_trade(self, trade: Trade, exit_price: float, time: datetime, reason: str):
        """Close a trade and calculate P&L."""
        # Apply slippage to exit
        if trade.action == 'LONG':
            exit_price *= (1 - self.slippage_percent / 100)
        else:
            exit_price *= (1 + self.slippage_percent / 100)

        trade.exit_time = time
        trade.exit_price = exit_price
        trade.exit_reason = reason

        # Calculate P&L
        if trade.action == 'LONG':
            trade.pnl_percent = ((exit_price - trade.entry_price) / trade.entry_price) * 100
        else:
            trade.pnl_percent = ((trade.entry_price - exit_price) / trade.entry_price) * 100

        # Apply commission
        trade.pnl_percent -= (self.commission_percent * 2)  # Entry + exit

        # Calculate position size and absolute P&L
        position_value = self.capital * (self.position_size_percent / 100)
        trade.pnl_absolute = position_value * (trade.pnl_percent / 100)

        # Update capital
        self.capital += trade.pnl_absolute

        # Calculate duration
        if isinstance(trade.entry_time, datetime) and isinstance(time, datetime):
            trade.duration_minutes = int((time - trade.entry_time).total_seconds() / 60)
        else:
            trade.duration_minutes = 0

        # Move from open to closed
        self.open_trades.remove(trade)
        self.trades.append(trade)

        logger.debug(f"Closed trade #{trade.trade_id} - {reason} - P&L: {trade.pnl_percent:.2f}%")

    def _calculate_equity(self, candle: pd.Series) -> float:
        """Calculate current equity including unrealized P&L."""
        equity = self.capital

        for trade in self.open_trades:
            position_value = self.initial_capital * (self.position_size_percent / 100)

            if trade.action == 'LONG':
                unrealized_pnl = ((candle['close'] - trade.entry_price) / trade.entry_price) * position_value
            else:
                unrealized_pnl = ((trade.entry_price - candle['close']) / trade.entry_price) * position_value

            equity += unrealized_pnl

        return equity

    def _calculate_results(self, strategy_name: str, symbol: str,
                          timeframe: str, start_date: datetime,
                          end_date: datetime) -> BacktestResult:
        """Calculate all backtest metrics."""

        total_trades = len(self.trades)

        if total_trades == 0:
            return BacktestResult(
                strategy_name=strategy_name,
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                initial_capital=self.initial_capital,
                final_capital=self.capital,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                expired_trades=0,
                win_rate=0.0,
                total_pnl_percent=0.0,
                total_pnl_absolute=0.0,
                average_win_percent=0.0,
                average_loss_percent=0.0,
                largest_win_percent=0.0,
                largest_loss_percent=0.0,
                max_drawdown_percent=0.0,
                max_consecutive_wins=0,
                max_consecutive_losses=0,
                profit_factor=0.0,
                sharpe_ratio=0.0,
                average_trade_duration_minutes=0.0,
                trades=[],
                equity_curve=self.equity_curve
            )

        # Categorize trades
        winning_trades = [t for t in self.trades if t.pnl_percent > 0]
        losing_trades = [t for t in self.trades if t.pnl_percent <= 0]
        expired_trades = [t for t in self.trades if t.exit_reason == 'EXPIRED']

        # Win rate
        win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0

        # P&L calculations
        total_pnl_percent = sum(t.pnl_percent for t in self.trades)
        total_pnl_absolute = self.capital - self.initial_capital

        avg_win = np.mean([t.pnl_percent for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl_percent for t in losing_trades]) if losing_trades else 0

        largest_win = max([t.pnl_percent for t in winning_trades]) if winning_trades else 0
        largest_loss = min([t.pnl_percent for t in losing_trades]) if losing_trades else 0

        # Profit factor
        gross_profit = sum(t.pnl_percent for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.pnl_percent for t in losing_trades)) if losing_trades else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit

        # Max drawdown
        max_drawdown = self._calculate_max_drawdown()

        # Consecutive wins/losses
        max_consec_wins, max_consec_losses = self._calculate_consecutive_streaks()

        # Sharpe ratio (simplified)
        sharpe = self._calculate_sharpe_ratio()

        # Average duration
        durations = [t.duration_minutes for t in self.trades if t.duration_minutes is not None]
        avg_duration = np.mean(durations) if durations else 0

        return BacktestResult(
            strategy_name=strategy_name,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=self.capital,
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            expired_trades=len(expired_trades),
            win_rate=win_rate,
            total_pnl_percent=total_pnl_percent,
            total_pnl_absolute=total_pnl_absolute,
            average_win_percent=avg_win,
            average_loss_percent=avg_loss,
            largest_win_percent=largest_win,
            largest_loss_percent=largest_loss,
            max_drawdown_percent=max_drawdown,
            max_consecutive_wins=max_consec_wins,
            max_consecutive_losses=max_consec_losses,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe,
            average_trade_duration_minutes=avg_duration,
            trades=self.trades,
            equity_curve=self.equity_curve
        )

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from equity curve."""
        if not self.equity_curve:
            return 0.0

        equities = [e[1] for e in self.equity_curve]
        peak = equities[0]
        max_dd = 0.0

        for equity in equities:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100
            if dd > max_dd:
                max_dd = dd

        return max_dd

    def _calculate_consecutive_streaks(self) -> Tuple[int, int]:
        """Calculate max consecutive wins and losses."""
        if not self.trades:
            return 0, 0

        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0

        for trade in self.trades:
            if trade.pnl_percent > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)

        return max_wins, max_losses

    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate simplified Sharpe ratio."""
        if not self.trades:
            return 0.0

        returns = [t.pnl_percent for t in self.trades]

        if len(returns) < 2:
            return 0.0

        avg_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0.0

        # Annualized (assuming 15m timeframe, ~35000 candles per year)
        sharpe = (avg_return - risk_free_rate / 35000) / std_return
        return sharpe * np.sqrt(35000)  # Annualized


class BacktestReporter:
    """Generate backtest reports."""

    @staticmethod
    def print_summary(result: BacktestResult):
        """Print summary to console."""
        print("\n" + "=" * 60)
        print(f"  BACKTEST RESULTS: {result.strategy_name}")
        print("=" * 60)
        print(f"\nSymbol: {result.symbol}")
        print(f"Timeframe: {result.timeframe}")
        print(f"Period: {result.start_date} to {result.end_date}")
        print(f"\nInitial Capital: ${result.initial_capital:,.2f}")
        print(f"Final Capital: ${result.final_capital:,.2f}")
        print(f"Total P&L: ${result.total_pnl_absolute:,.2f} ({result.total_pnl_percent:.2f}%)")

        print("\n--- TRADE STATISTICS ---")
        print(f"Total Trades: {result.total_trades}")
        print(f"Winning Trades: {result.winning_trades}")
        print(f"Losing Trades: {result.losing_trades}")
        print(f"Win Rate: {result.win_rate:.2f}%")

        print("\n--- PERFORMANCE METRICS ---")
        print(f"Average Win: {result.average_win_percent:.2f}%")
        print(f"Average Loss: {result.average_loss_percent:.2f}%")
        print(f"Largest Win: {result.largest_win_percent:.2f}%")
        print(f"Largest Loss: {result.largest_loss_percent:.2f}%")
        print(f"Profit Factor: {result.profit_factor:.2f}")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")

        print("\n--- RISK METRICS ---")
        print(f"Max Drawdown: {result.max_drawdown_percent:.2f}%")
        print(f"Max Consecutive Wins: {result.max_consecutive_wins}")
        print(f"Max Consecutive Losses: {result.max_consecutive_losses}")

        print("\n--- TIME METRICS ---")
        print(f"Average Trade Duration: {result.average_trade_duration_minutes:.0f} minutes")

        print("\n" + "=" * 60)

    @staticmethod
    def to_dict(result: BacktestResult) -> Dict:
        """Convert result to dictionary for JSON export."""
        return {
            'strategy_name': result.strategy_name,
            'symbol': result.symbol,
            'timeframe': result.timeframe,
            'start_date': str(result.start_date),
            'end_date': str(result.end_date),
            'initial_capital': result.initial_capital,
            'final_capital': result.final_capital,
            'total_trades': result.total_trades,
            'winning_trades': result.winning_trades,
            'losing_trades': result.losing_trades,
            'win_rate': result.win_rate,
            'total_pnl_percent': result.total_pnl_percent,
            'total_pnl_absolute': result.total_pnl_absolute,
            'average_win_percent': result.average_win_percent,
            'average_loss_percent': result.average_loss_percent,
            'largest_win_percent': result.largest_win_percent,
            'largest_loss_percent': result.largest_loss_percent,
            'max_drawdown_percent': result.max_drawdown_percent,
            'max_consecutive_wins': result.max_consecutive_wins,
            'max_consecutive_losses': result.max_consecutive_losses,
            'profit_factor': result.profit_factor,
            'sharpe_ratio': result.sharpe_ratio,
            'average_trade_duration_minutes': result.average_trade_duration_minutes
        }

    @staticmethod
    def save_json(result: BacktestResult, filepath: str):
        """Save result to JSON file."""
        data = BacktestReporter.to_dict(result)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to {filepath}")

    @staticmethod
    def compare_results(results: List[BacktestResult]):
        """Compare multiple backtest results."""
        print("\n" + "=" * 80)
        print("  BACKTEST COMPARISON")
        print("=" * 80)

        headers = ['Strategy', 'Trades', 'Win Rate', 'Total P&L', 'Profit Factor', 'Max DD', 'Sharpe']

        print(f"\n{'Strategy':<25} {'Trades':>8} {'Win Rate':>10} {'P&L %':>10} {'PF':>8} {'Max DD':>10} {'Sharpe':>8}")
        print("-" * 80)

        for r in results:
            print(f"{r.strategy_name:<25} {r.total_trades:>8} {r.win_rate:>9.1f}% {r.total_pnl_percent:>9.1f}% {r.profit_factor:>8.2f} {r.max_drawdown_percent:>9.1f}% {r.sharpe_ratio:>8.2f}")

        print("\n" + "=" * 80)

        # Find best performer
        if results:
            best_winrate = max(results, key=lambda x: x.win_rate)
            best_pnl = max(results, key=lambda x: x.total_pnl_percent)
            best_sharpe = max(results, key=lambda x: x.sharpe_ratio)

            print(f"\nBest Win Rate: {best_winrate.strategy_name} ({best_winrate.win_rate:.1f}%)")
            print(f"Best P&L: {best_pnl.strategy_name} ({best_pnl.total_pnl_percent:.1f}%)")
            print(f"Best Sharpe: {best_sharpe.strategy_name} ({best_sharpe.sharpe_ratio:.2f})")
