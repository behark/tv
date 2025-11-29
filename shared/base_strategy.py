#!/usr/bin/env python3
"""
Base Strategy Class
Provides common strategy functionality for all tiers.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, List
from abc import ABC, abstractmethod
from datetime import datetime

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    Each tier inherits from this and implements tier-specific logic.
    """

    def __init__(self, config: dict):
        self.config = config
        self.last_bar_time = {}
        self.last_signal = {}
        self.strategy_name = "base"

        # Common settings
        self.enable_long = config.get('enable_long', True)
        self.enable_short = config.get('enable_short', True)
        self.score_threshold = config.get('strategy', {}).get('score_threshold', 3.0)
        self.risk_reward_ratio = config.get('strategy', {}).get('risk_reward_ratio', 1.5)
        self.atr_multiplier_sl = config.get('strategy', {}).get('atr_multiplier_sl', 1.5)

    @abstractmethod
    def calculate_signal_score(self, action: str, last_bar: pd.Series,
                                prev_bar: pd.Series) -> tuple:
        """
        Calculate signal score based on tier-specific indicators.
        Must be implemented by subclass.
        Returns: (score, indicators_dict)
        """
        pass

    @abstractmethod
    def get_weights(self) -> dict:
        """Get indicator weights for scoring. Must be implemented by subclass."""
        pass

    def check_signals(self, symbol: str, df: pd.DataFrame) -> Optional[Dict]:
        """Main signal checking function."""
        if df is None or len(df) < 2:
            return None

        try:
            last_bar = df.iloc[-1]
            prev_bar = df.iloc[-2]
            bar_time = df.index[-1]

            # Avoid duplicate signals on same bar
            if symbol in self.last_bar_time and self.last_bar_time[symbol] == bar_time:
                return None

            self.last_bar_time[symbol] = bar_time

            # Calculate scores for both directions
            long_score, long_indicators = self.calculate_signal_score('LONG', last_bar, prev_bar)
            short_score, short_indicators = self.calculate_signal_score('SHORT', last_bar, prev_bar)

            signal = None

            # Check LONG signal
            if self.enable_long and long_score >= self.score_threshold:
                signal_key = f"{symbol}_LONG"
                if signal_key not in self.last_signal or self.last_signal[signal_key] != bar_time:
                    self.last_signal[signal_key] = bar_time
                    signal = self._create_signal(
                        symbol, 'LONG', last_bar, long_score, long_indicators, bar_time
                    )

            # Check SHORT signal (if no LONG signal)
            elif self.enable_short and short_score >= self.score_threshold:
                signal_key = f"{symbol}_SHORT"
                if signal_key not in self.last_signal or self.last_signal[signal_key] != bar_time:
                    self.last_signal[signal_key] = bar_time
                    signal = self._create_signal(
                        symbol, 'SHORT', last_bar, short_score, short_indicators, bar_time
                    )

            if signal:
                logger.info(f"{signal['action']} signal for {symbol} at {signal['entry_price']:.6f} "
                           f"(score: {signal['score']:.1f}, strategy: {self.strategy_name})")

            return signal

        except Exception as e:
            logger.error(f"Error checking signals for {symbol}: {e}")
            return None

    def _create_signal(self, symbol: str, action: str, last_bar: pd.Series,
                       score: float, indicators: dict, bar_time) -> Dict:
        """Create a signal dictionary."""
        entry_price = last_bar['close']
        atr = last_bar.get('atr', entry_price * 0.02)  # Default 2% if no ATR

        if action == 'LONG':
            stop_loss = entry_price - (atr * self.atr_multiplier_sl)
            take_profit = entry_price + (atr * self.atr_multiplier_sl * self.risk_reward_ratio)
        else:
            stop_loss = entry_price + (atr * self.atr_multiplier_sl)
            take_profit = entry_price - (atr * self.atr_multiplier_sl * self.risk_reward_ratio)

        return {
            'symbol': symbol,
            'action': action,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'score': score,
            'max_score': sum(self.get_weights().values()),
            'indicators': indicators,
            'strategy_name': self.strategy_name,
            'timestamp': bar_time,
            'atr': atr,
            'adx': last_bar.get('adx', 0),
            'rsi': last_bar.get('rsi', last_bar.get('rsx', 50)),
            'volume_ratio': last_bar.get('volume_ratio', 1.0)
        }

    def calculate_dynamic_stops(self, action: str, entry_price: float,
                                 atr: float, last_bar: pd.Series) -> tuple:
        """
        Calculate dynamic stop loss and take profit levels.
        Can use various methods: ATR, Chandelier, Parabolic SAR, etc.
        """
        # Default ATR-based stops
        if action == 'LONG':
            stop_loss = entry_price - (atr * self.atr_multiplier_sl)
            take_profit = entry_price + (atr * self.atr_multiplier_sl * self.risk_reward_ratio)

            # Use Chandelier Exit if available
            if 'ce_long' in last_bar and last_bar['ce_long'] > 0:
                stop_loss = max(stop_loss, last_bar['ce_long'])

        else:  # SHORT
            stop_loss = entry_price + (atr * self.atr_multiplier_sl)
            take_profit = entry_price - (atr * self.atr_multiplier_sl * self.risk_reward_ratio)

            if 'ce_short' in last_bar and last_bar['ce_short'] > 0:
                stop_loss = min(stop_loss, last_bar['ce_short'])

        return stop_loss, take_profit

    def is_trending(self, last_bar: pd.Series, threshold: float = 25) -> bool:
        """Check if market is trending based on ADX."""
        adx = last_bar.get('adx', 0)
        return adx >= threshold

    def is_ranging(self, last_bar: pd.Series, threshold: float = 20) -> bool:
        """Check if market is ranging based on ADX."""
        adx = last_bar.get('adx', 100)
        return adx < threshold

    def get_trend_direction(self, last_bar: pd.Series) -> str:
        """Determine trend direction from multiple indicators."""
        bullish_count = 0
        bearish_count = 0

        close = last_bar['close']

        # Check EMA/FRAMA
        for ma_col in ['ema', 'frama', 'vwap']:
            if ma_col in last_bar:
                if close > last_bar[ma_col]:
                    bullish_count += 1
                else:
                    bearish_count += 1

        # Check SuperTrend
        if 'supertrend_dir' in last_bar:
            if last_bar['supertrend_dir'] == 1:
                bullish_count += 1
            else:
                bearish_count += 1

        # Check MACD
        if 'macd' in last_bar and 'macd_signal' in last_bar:
            if last_bar['macd'] > last_bar['macd_signal']:
                bullish_count += 1
            else:
                bearish_count += 1

        if bullish_count > bearish_count:
            return 'LONG'
        elif bearish_count > bullish_count:
            return 'SHORT'
        else:
            return 'NEUTRAL'

    def check_momentum(self, action: str, last_bar: pd.Series, prev_bar: pd.Series) -> bool:
        """Check if momentum supports the trade direction."""
        # RSI/RSX check
        for rsi_col in ['rsi', 'rsx', 'stoch_rsi_k']:
            if rsi_col in last_bar:
                rsi = last_bar[rsi_col]
                if action == 'LONG':
                    return 30 <= rsi <= 70  # Not overbought
                else:
                    return 30 <= rsi <= 70  # Not oversold

        return True  # Default allow if no RSI available

    def check_volume_confirmation(self, last_bar: pd.Series,
                                   min_ratio: float = 1.0) -> bool:
        """Check if volume confirms the move."""
        if 'volume_ratio' in last_bar:
            return last_bar['volume_ratio'] >= min_ratio
        return True
