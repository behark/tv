#!/usr/bin/env python3
"""
Tier 4 - High Win Rate Indicators
Multi-confluence mean reversion with trend alignment for 75-85% accuracy.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from base_indicators import BaseIndicatorCalculator


class Tier4Indicators(BaseIndicatorCalculator):
    """
    High Win Rate Indicator Suite

    Core Concepts:
    1. Mean Reversion - Trade back to mean from extremes
    2. Multi-Timeframe Trend - Only trade with higher TF trend
    3. Volume Confirmation - Require volume spike on reversals
    4. Market Regime - Identify trending vs ranging conditions
    5. Confluence Zones - Multiple indicators agreeing
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.indicator_config = config.get('indicators', {})

        # RSI settings
        self.rsi_period = self.indicator_config.get('rsi_period', 14)
        self.rsi_oversold = self.indicator_config.get('rsi_oversold', 30)
        self.rsi_overbought = self.indicator_config.get('rsi_overbought', 70)
        self.rsi_extreme_oversold = self.indicator_config.get('rsi_extreme_oversold', 20)
        self.rsi_extreme_overbought = self.indicator_config.get('rsi_extreme_overbought', 80)

        # Bollinger Bands
        self.bb_period = self.indicator_config.get('bb_period', 20)
        self.bb_std = self.indicator_config.get('bb_std', 2.0)

        # Moving averages for trend
        self.ema_fast = self.indicator_config.get('ema_fast', 21)
        self.ema_slow = self.indicator_config.get('ema_slow', 50)
        self.ema_trend = self.indicator_config.get('ema_trend', 200)

        # Stochastic RSI
        self.stoch_period = self.indicator_config.get('stoch_period', 14)
        self.stoch_smooth = self.indicator_config.get('stoch_smooth', 3)

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all Tier 4 indicators."""
        if df is None or len(df) < 50:
            return None

        df = df.copy()

        # Core trend indicators
        df = self.calculate_ema(df, self.ema_fast, f'ema_{self.ema_fast}')
        df = self.calculate_ema(df, self.ema_slow, f'ema_{self.ema_slow}')
        df = self.calculate_ema(df, self.ema_trend, f'ema_{self.ema_trend}')

        # Mean reversion indicators
        df = self.calculate_rsi(df, self.rsi_period)
        df = self.calculate_bollinger_bands(df, self.bb_period, self.bb_std)
        df = self._calculate_stochastic_rsi(df)

        # VWAP for intraday mean
        df = self.calculate_vwap(df)

        # Volume analysis
        df = self._calculate_volume_analysis(df)

        # Market regime detection
        df = self._calculate_market_regime(df)

        # ATR for stops
        df = self.calculate_atr(df)

        # Confluence score
        df = self._calculate_confluence_score(df)

        # RSI divergence
        df = self._calculate_rsi_divergence(df)

        # Support/Resistance levels
        df = self._calculate_sr_levels(df)

        # Mean reversion signals
        df = self._calculate_mean_reversion_signals(df)

        return df

    def _calculate_stochastic_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Stochastic RSI."""
        rsi = df['rsi'].values
        period = self.stoch_period
        smooth = self.stoch_smooth

        stoch_rsi = np.full(len(rsi), np.nan)

        for i in range(period, len(rsi)):
            rsi_window = rsi[i-period+1:i+1]
            rsi_min = np.nanmin(rsi_window)
            rsi_max = np.nanmax(rsi_window)

            if rsi_max - rsi_min > 0:
                stoch_rsi[i] = (rsi[i] - rsi_min) / (rsi_max - rsi_min) * 100
            else:
                stoch_rsi[i] = 50

        # Smooth with SMA
        df['stoch_rsi_k'] = pd.Series(stoch_rsi).rolling(smooth).mean().values
        df['stoch_rsi_d'] = df['stoch_rsi_k'].rolling(smooth).mean()

        return df

    def _calculate_volume_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based indicators."""
        # Volume SMA
        df['volume_sma'] = df['volume'].rolling(20).mean()

        # Volume ratio
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        # Volume spike detection (1.5x average)
        df['volume_spike'] = df['volume_ratio'] > 1.5

        # Climax volume (2.5x average) - potential reversal
        df['volume_climax'] = df['volume_ratio'] > 2.5

        # On-Balance Volume
        obv = np.zeros(len(df))
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv[i] = obv[i-1] + df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv[i] = obv[i-1] - df['volume'].iloc[i]
            else:
                obv[i] = obv[i-1]

        df['obv'] = obv
        df['obv_sma'] = pd.Series(obv).rolling(20).mean().values

        # OBV trend
        df['obv_rising'] = df['obv'] > df['obv_sma']

        return df

    def _calculate_market_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect market regime: trending vs ranging.
        Only take mean reversion trades in ranging markets.
        """
        # ADX for trend strength
        df = self.calculate_adx(df, 14)

        # Bollinger Band width for volatility
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'] * 100
        df['bb_width_sma'] = df['bb_width'].rolling(20).mean()

        # Regime classification
        # Ranging: ADX < 25 and BB width below average
        # Trending: ADX > 25
        df['regime_trending'] = df['adx'] > 25
        df['regime_ranging'] = (df['adx'] < 25) & (df['bb_width'] < df['bb_width_sma'])

        # Trend direction when trending
        ema_fast_col = f'ema_{self.ema_fast}'
        ema_slow_col = f'ema_{self.ema_slow}'
        ema_trend_col = f'ema_{self.ema_trend}'

        df['trend_up'] = (df[ema_fast_col] > df[ema_slow_col]) & (df['close'] > df[ema_trend_col])
        df['trend_down'] = (df[ema_fast_col] < df[ema_slow_col]) & (df['close'] < df[ema_trend_col])

        # Price position relative to EMAs
        df['above_all_emas'] = (df['close'] > df[ema_fast_col]) & \
                               (df['close'] > df[ema_slow_col]) & \
                               (df['close'] > df[ema_trend_col])
        df['below_all_emas'] = (df['close'] < df[ema_fast_col]) & \
                               (df['close'] < df[ema_slow_col]) & \
                               (df['close'] < df[ema_trend_col])

        return df

    def _calculate_confluence_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate confluence score for entries.
        Higher score = more confirmations = higher probability.
        """
        bullish_score = np.zeros(len(df))
        bearish_score = np.zeros(len(df))

        for i in range(1, len(df)):
            bull = 0
            bear = 0

            # RSI conditions
            if df['rsi'].iloc[i] < self.rsi_oversold:
                bull += 1
            if df['rsi'].iloc[i] < self.rsi_extreme_oversold:
                bull += 1  # Extra point for extreme
            if df['rsi'].iloc[i] > self.rsi_overbought:
                bear += 1
            if df['rsi'].iloc[i] > self.rsi_extreme_overbought:
                bear += 1

            # Stochastic RSI
            if df['stoch_rsi_k'].iloc[i] < 20:
                bull += 1
            if df['stoch_rsi_k'].iloc[i] > 80:
                bear += 1

            # Bollinger Band touch
            if df['close'].iloc[i] <= df['bb_lower'].iloc[i]:
                bull += 1.5
            if df['close'].iloc[i] >= df['bb_upper'].iloc[i]:
                bear += 1.5

            # VWAP position (if available)
            if pd.notna(df['vwap'].iloc[i]):
                vwap_dist = (df['close'].iloc[i] - df['vwap'].iloc[i]) / df['vwap'].iloc[i] * 100
                if vwap_dist < -1:  # More than 1% below VWAP
                    bull += 1
                if vwap_dist > 1:   # More than 1% above VWAP
                    bear += 1

            # Volume confirmation
            if df['volume_spike'].iloc[i]:
                bull += 0.5
                bear += 0.5
            if df['volume_climax'].iloc[i]:
                bull += 1
                bear += 1

            # Trend alignment (bonus for trading with trend)
            if df['trend_up'].iloc[i]:
                bull += 1
            if df['trend_down'].iloc[i]:
                bear += 1

            bullish_score[i] = bull
            bearish_score[i] = bear

        df['bullish_confluence'] = bullish_score
        df['bearish_confluence'] = bearish_score

        return df

    def _calculate_rsi_divergence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect RSI divergence for reversal confirmation."""
        lookback = 14

        bullish_div = np.zeros(len(df), dtype=bool)
        bearish_div = np.zeros(len(df), dtype=bool)

        for i in range(lookback, len(df)):
            # Find swing lows in price
            price_window = df['low'].iloc[i-lookback:i+1].values
            rsi_window = df['rsi'].iloc[i-lookback:i+1].values

            # Bullish divergence: lower low in price, higher low in RSI
            price_min_idx = np.argmin(price_window)
            if price_min_idx == lookback:  # Current bar is lowest
                # Find previous low
                prev_price_window = df['low'].iloc[i-lookback*2:i-lookback+1].values
                if len(prev_price_window) > 0:
                    prev_min_idx = np.argmin(prev_price_window)
                    prev_price_low = prev_price_window[prev_min_idx]
                    prev_rsi_low = df['rsi'].iloc[i-lookback*2+prev_min_idx]

                    curr_price_low = df['low'].iloc[i]
                    curr_rsi_low = df['rsi'].iloc[i]

                    if curr_price_low < prev_price_low and curr_rsi_low > prev_rsi_low:
                        bullish_div[i] = True

            # Bearish divergence: higher high in price, lower high in RSI
            price_window_high = df['high'].iloc[i-lookback:i+1].values
            price_max_idx = np.argmax(price_window_high)
            if price_max_idx == lookback:
                prev_price_window = df['high'].iloc[i-lookback*2:i-lookback+1].values
                if len(prev_price_window) > 0:
                    prev_max_idx = np.argmax(prev_price_window)
                    prev_price_high = prev_price_window[prev_max_idx]
                    prev_rsi_high = df['rsi'].iloc[i-lookback*2+prev_max_idx]

                    curr_price_high = df['high'].iloc[i]
                    curr_rsi_high = df['rsi'].iloc[i]

                    if curr_price_high > prev_price_high and curr_rsi_high < prev_rsi_high:
                        bearish_div[i] = True

        df['rsi_bullish_div'] = bullish_div
        df['rsi_bearish_div'] = bearish_div

        return df

    def _calculate_sr_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate support and resistance levels."""
        lookback = 20

        # Find recent swing highs and lows
        swing_highs = []
        swing_lows = []

        for i in range(lookback, len(df) - lookback):
            # Swing high: highest in window
            if df['high'].iloc[i] == df['high'].iloc[i-lookback:i+lookback+1].max():
                swing_highs.append(df['high'].iloc[i])

            # Swing low: lowest in window
            if df['low'].iloc[i] == df['low'].iloc[i-lookback:i+lookback+1].min():
                swing_lows.append(df['low'].iloc[i])

        # Get nearest support and resistance
        nearest_support = np.full(len(df), np.nan)
        nearest_resistance = np.full(len(df), np.nan)

        for i in range(len(df)):
            current_price = df['close'].iloc[i]

            # Nearest support below current price
            supports_below = [s for s in swing_lows if s < current_price]
            if supports_below:
                nearest_support[i] = max(supports_below)

            # Nearest resistance above current price
            resistances_above = [r for r in swing_highs if r > current_price]
            if resistances_above:
                nearest_resistance[i] = min(resistances_above)

        df['nearest_support'] = nearest_support
        df['nearest_resistance'] = nearest_resistance

        # Distance to S/R as percentage
        df['dist_to_support'] = (df['close'] - df['nearest_support']) / df['close'] * 100
        df['dist_to_resistance'] = (df['nearest_resistance'] - df['close']) / df['close'] * 100

        # Near support/resistance (within 0.5%)
        df['near_support'] = df['dist_to_support'] < 0.5
        df['near_resistance'] = df['dist_to_resistance'] < 0.5

        return df

    def _calculate_mean_reversion_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate mean reversion signal strength.
        Combines all confluence factors.
        """
        mr_bull_signal = np.zeros(len(df))
        mr_bear_signal = np.zeros(len(df))

        for i in range(1, len(df)):
            bull_strength = 0
            bear_strength = 0

            # Base confluence score
            bull_strength += df['bullish_confluence'].iloc[i]
            bear_strength += df['bearish_confluence'].iloc[i]

            # RSI divergence bonus
            if df['rsi_bullish_div'].iloc[i]:
                bull_strength += 2
            if df['rsi_bearish_div'].iloc[i]:
                bear_strength += 2

            # Near S/R bonus
            if df['near_support'].iloc[i]:
                bull_strength += 1.5
            if df['near_resistance'].iloc[i]:
                bear_strength += 1.5

            # Regime filter (prefer ranging for mean reversion)
            if df['regime_ranging'].iloc[i]:
                bull_strength *= 1.2
                bear_strength *= 1.2

            # Penalize counter-trend trades
            if df['trend_down'].iloc[i] and bull_strength > 0:
                bull_strength *= 0.7  # Reduce bullish signal in downtrend
            if df['trend_up'].iloc[i] and bear_strength > 0:
                bear_strength *= 0.7  # Reduce bearish signal in uptrend

            mr_bull_signal[i] = bull_strength
            mr_bear_signal[i] = bear_strength

        df['mr_bullish_signal'] = mr_bull_signal
        df['mr_bearish_signal'] = mr_bear_signal

        return df
