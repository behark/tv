#!/usr/bin/env python3
"""
Base Indicator Calculator
Provides common indicator calculation methods for all tiers.
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
import logging
from typing import Dict, Tuple, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseIndicatorCalculator(ABC):
    """
    Abstract base class for indicator calculators.
    Each tier inherits from this and implements tier-specific indicators.
    """

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def calculate_tier_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate tier-specific indicators. Must be implemented by subclass."""
        pass

    def calculate_common_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate common indicators used by all tiers."""
        strategy = self.config.get('strategy', {})

        # ATR - Always needed for stops
        atr_length = strategy.get('atr_length', 14)
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=atr_length)

        # Volume average
        volume_period = strategy.get('volume_period', 20)
        df['avg_volume'] = ta.sma(df['volume'], length=volume_period)
        df['volume_ratio'] = df['volume'] / df['avg_volume']

        # ADX for trend strength
        adx_length = strategy.get('adx_length', 14)
        adx_result = ta.adx(df['high'], df['low'], df['close'], length=adx_length)
        df['adx'] = adx_result[[c for c in adx_result.columns if c.startswith('ADX_')][0]]
        df['di_plus'] = adx_result[[c for c in adx_result.columns if c.startswith('DMP_')][0]]
        df['di_minus'] = adx_result[[c for c in adx_result.columns if c.startswith('DMN_')][0]]

        # MACD
        macd_fast = strategy.get('macd_fast', 12)
        macd_slow = strategy.get('macd_slow', 26)
        macd_signal = strategy.get('macd_signal', 9)
        macd_result = ta.macd(df['close'], fast=macd_fast, slow=macd_slow, signal=macd_signal)
        macd_col = [c for c in macd_result.columns if c.startswith('MACD_') and 'h' not in c.lower() and 's' not in c.lower()][0]
        signal_col = [c for c in macd_result.columns if c.startswith('MACDs_')][0]
        hist_col = [c for c in macd_result.columns if c.startswith('MACDh_')][0]
        df['macd'] = macd_result[macd_col]
        df['macd_signal'] = macd_result[signal_col]
        df['macd_hist'] = macd_result[hist_col]

        return df

    def calculate_all(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculate all indicators for the tier."""
        if df is None or len(df) < 200:
            logger.warning("Insufficient data for indicator calculation")
            return None

        df = df.copy()

        try:
            # Common indicators first
            df = self.calculate_common_indicators(df)

            # Tier-specific indicators
            df = self.calculate_tier_indicators(df)

            # Drop NaN values
            df.dropna(inplace=True)

            if len(df) < 50:
                logger.warning(f"Insufficient data after calculation: {len(df)} bars")
                return None

            logger.debug(f"Calculated all indicators for {len(df)} bars")
            return df

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return None

    # ==================== Common Indicator Methods ====================

    def calculate_ema(self, df: pd.DataFrame, length: int, column: str = 'close') -> pd.Series:
        """Calculate EMA"""
        return ta.ema(df[column], length=length)

    def calculate_sma(self, df: pd.DataFrame, length: int, column: str = 'close') -> pd.Series:
        """Calculate SMA"""
        return ta.sma(df[column], length=length)

    def calculate_rsi(self, df: pd.DataFrame, length: int = 14) -> pd.Series:
        """Calculate RSI"""
        return ta.rsi(df['close'], length=length)

    def calculate_supertrend(self, df: pd.DataFrame, length: int = 22,
                              mult: float = 3.0) -> Tuple[pd.Series, pd.Series]:
        """Calculate SuperTrend. Returns (supertrend, direction)"""
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values

            atr = ta.atr(df['high'], df['low'], df['close'], length=length).values

            hl2 = (high + low) / 2
            basic_ub = hl2 + mult * atr
            basic_lb = hl2 - mult * atr

            final_ub = np.zeros(len(close))
            final_lb = np.zeros(len(close))
            supertrend = np.zeros(len(close))
            direction = np.ones(len(close))

            final_ub[0] = basic_ub[0]
            final_lb[0] = basic_lb[0]
            supertrend[0] = final_ub[0]
            direction[0] = -1

            for i in range(1, len(close)):
                if np.isnan(basic_ub[i]) or np.isnan(final_ub[i-1]):
                    final_ub[i] = basic_ub[i] if not np.isnan(basic_ub[i]) else final_ub[i-1]
                elif basic_ub[i] < final_ub[i-1] or close[i-1] > final_ub[i-1]:
                    final_ub[i] = basic_ub[i]
                else:
                    final_ub[i] = final_ub[i-1]

                if np.isnan(basic_lb[i]) or np.isnan(final_lb[i-1]):
                    final_lb[i] = basic_lb[i] if not np.isnan(basic_lb[i]) else final_lb[i-1]
                elif basic_lb[i] > final_lb[i-1] or close[i-1] < final_lb[i-1]:
                    final_lb[i] = basic_lb[i]
                else:
                    final_lb[i] = final_lb[i-1]

                if supertrend[i-1] == final_ub[i-1]:
                    if close[i] <= final_ub[i]:
                        supertrend[i] = final_ub[i]
                        direction[i] = -1
                    else:
                        supertrend[i] = final_lb[i]
                        direction[i] = 1
                else:
                    if close[i] >= final_lb[i]:
                        supertrend[i] = final_lb[i]
                        direction[i] = 1
                    else:
                        supertrend[i] = final_ub[i]
                        direction[i] = -1

            return pd.Series(supertrend, index=df.index), pd.Series(direction, index=df.index)

        except Exception as e:
            logger.error(f"Error calculating SuperTrend: {e}")
            return pd.Series(np.nan, index=df.index), pd.Series(0, index=df.index)

    def calculate_bollinger_bands(self, df: pd.DataFrame, length: int = 20,
                                   std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands. Returns (upper, middle, lower)"""
        bb = ta.bbands(df['close'], length=length, std=std)
        upper = bb[[c for c in bb.columns if 'BBU' in c][0]]
        middle = bb[[c for c in bb.columns if 'BBM' in c][0]]
        lower = bb[[c for c in bb.columns if 'BBL' in c][0]]
        return upper, middle, lower

    def calculate_keltner_channels(self, df: pd.DataFrame, length: int = 20,
                                    mult: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Keltner Channels. Returns (upper, basis, lower)"""
        kc = ta.kc(df['high'], df['low'], df['close'], length=length, scalar=mult)
        upper = kc[[c for c in kc.columns if 'KCU' in c][0]]
        basis = kc[[c for c in kc.columns if 'KCB' in c][0]]
        lower = kc[[c for c in kc.columns if 'KCL' in c][0]]
        return upper, basis, lower

    def calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate VWAP"""
        return ta.vwap(df['high'], df['low'], df['close'], df['volume'])

    def calculate_stochastic_rsi(self, df: pd.DataFrame, length: int = 14,
                                  rsi_length: int = 14, k: int = 3,
                                  d: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic RSI. Returns (K, D)"""
        stoch_rsi = ta.stochrsi(df['close'], length=length, rsi_length=rsi_length, k=k, d=d)
        k_line = stoch_rsi[[c for c in stoch_rsi.columns if 'STOCHRSIk' in c][0]]
        d_line = stoch_rsi[[c for c in stoch_rsi.columns if 'STOCHRSId' in c][0]]
        return k_line, d_line

    def calculate_ichimoku(self, df: pd.DataFrame, tenkan: int = 9, kijun: int = 26,
                           senkou: int = 52) -> Dict[str, pd.Series]:
        """Calculate Ichimoku Cloud components"""
        ichimoku = ta.ichimoku(df['high'], df['low'], df['close'],
                               tenkan=tenkan, kijun=kijun, senkou=senkou)

        result = {}
        for col in ichimoku[0].columns:
            if 'ISA' in col:
                result['senkou_a'] = ichimoku[0][col]
            elif 'ISB' in col:
                result['senkou_b'] = ichimoku[0][col]
            elif 'ITS' in col:
                result['tenkan'] = ichimoku[0][col]
            elif 'IKS' in col:
                result['kijun'] = ichimoku[0][col]

        return result

    def calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On Balance Volume"""
        return ta.obv(df['close'], df['volume'])

    def calculate_cmf(self, df: pd.DataFrame, length: int = 20) -> pd.Series:
        """Calculate Chaikin Money Flow"""
        return ta.cmf(df['high'], df['low'], df['close'], df['volume'], length=length)

    def calculate_mfi(self, df: pd.DataFrame, length: int = 14) -> pd.Series:
        """Calculate Money Flow Index"""
        return ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=length)

    def calculate_volatility_ratio(self, df: pd.DataFrame, short_length: int = 5,
                                    long_length: int = 20) -> pd.Series:
        """Calculate Volatility Ratio"""
        high = df['high'].values
        low = df['low'].values

        vr = np.zeros(len(high))

        for i in range(long_length, len(high)):
            short_range = np.max(high[i-short_length+1:i+1]) - np.min(low[i-short_length+1:i+1])
            long_range = np.max(high[i-long_length+1:i+1]) - np.min(low[i-long_length+1:i+1])

            if long_range != 0:
                vr[i] = short_range / long_range
            else:
                vr[i] = 1.0

        return pd.Series(vr, index=df.index)

    def calculate_cvd(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Cumulative Volume Delta (simplified version using candle analysis).
        Positive delta = buying pressure, Negative delta = selling pressure.
        """
        # Estimate buy/sell volume from candle structure
        body = df['close'] - df['open']
        total_range = df['high'] - df['low']

        # Avoid division by zero
        total_range = total_range.replace(0, np.nan)

        # Estimate: volume * (body position in range)
        # If close > open (bullish candle), more volume is "buy"
        buy_ratio = np.where(
            body >= 0,
            0.5 + (body / total_range) * 0.5,  # Bullish: 50-100% buy
            0.5 + (body / total_range) * 0.5   # Bearish: 0-50% buy
        )

        buy_volume = df['volume'] * buy_ratio
        sell_volume = df['volume'] * (1 - buy_ratio)

        delta = buy_volume - sell_volume
        cvd = delta.cumsum()

        return cvd

    def calculate_market_structure(self, df: pd.DataFrame,
                                    lookback: int = 5) -> Dict[str, pd.Series]:
        """
        Detect market structure: Higher Highs, Higher Lows, Lower Highs, Lower Lows.
        Returns dict with swing points and structure.
        """
        high = df['high'].values
        low = df['low'].values

        swing_high = np.zeros(len(high))
        swing_low = np.zeros(len(low))
        structure = np.zeros(len(high))  # 1 = bullish, -1 = bearish, 0 = neutral

        for i in range(lookback, len(high) - lookback):
            # Swing High: highest point in lookback window
            if high[i] == max(high[i-lookback:i+lookback+1]):
                swing_high[i] = high[i]

            # Swing Low: lowest point in lookback window
            if low[i] == min(low[i-lookback:i+lookback+1]):
                swing_low[i] = low[i]

        # Determine structure
        last_swing_high = 0
        last_swing_low = 0

        for i in range(len(high)):
            if swing_high[i] > 0:
                if last_swing_high > 0:
                    if swing_high[i] > last_swing_high:
                        structure[i] = 1  # Higher High
                    else:
                        structure[i] = -1  # Lower High
                last_swing_high = swing_high[i]

            if swing_low[i] > 0:
                if last_swing_low > 0:
                    if swing_low[i] > last_swing_low:
                        structure[i] = max(structure[i], 1)  # Higher Low
                    else:
                        structure[i] = min(structure[i], -1)  # Lower Low
                last_swing_low = swing_low[i]

        return {
            'swing_high': pd.Series(swing_high, index=df.index),
            'swing_low': pd.Series(swing_low, index=df.index),
            'structure': pd.Series(structure, index=df.index)
        }
