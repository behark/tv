#!/usr/bin/env python3
"""
Tier 2 - Advanced Technical Indicators
Focus: Stochastic RSI, Keltner Channels, Fisher Transform, Ehlers, Waddah Attar

These are sophisticated technical indicators that go beyond basic TA.
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
import logging
from typing import Dict, Tuple, Optional
import sys
sys.path.insert(0, '../shared')

from base_indicators import BaseIndicatorCalculator

logger = logging.getLogger(__name__)


class Tier2Indicators(BaseIndicatorCalculator):
    """
    Tier 2 Advanced Technical Indicators:
    1. Stochastic RSI - More sensitive momentum
    2. Keltner Channels - ATR-based volatility bands
    3. Fisher Transform - Gaussian price transformation
    4. Ehlers MESA - Adaptive cycle analysis
    5. Waddah Attar Explosion - Entry timing
    6. Half Trend - Clean trend signals
    7. FRAMA - Fractal Adaptive MA
    8. Squeeze Momentum - Volatility + Momentum combo
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.strategy = config.get('strategy', {})

    def calculate_tier_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Tier 2 specific indicators."""

        # 1. Stochastic RSI
        stoch_length = self.strategy.get('stoch_rsi_length', 14)
        stoch_k, stoch_d = self.calculate_stochastic_rsi(df, length=stoch_length)
        df['stoch_rsi_k'] = stoch_k
        df['stoch_rsi_d'] = stoch_d

        # StochRSI signals
        df['stoch_rsi_oversold'] = df['stoch_rsi_k'] < 20
        df['stoch_rsi_overbought'] = df['stoch_rsi_k'] > 80
        df['stoch_rsi_cross_up'] = (df['stoch_rsi_k'] > df['stoch_rsi_d']) & (df['stoch_rsi_k'].shift(1) <= df['stoch_rsi_d'].shift(1))
        df['stoch_rsi_cross_down'] = (df['stoch_rsi_k'] < df['stoch_rsi_d']) & (df['stoch_rsi_k'].shift(1) >= df['stoch_rsi_d'].shift(1))

        # 2. Keltner Channels
        kc_length = self.strategy.get('keltner_length', 20)
        kc_mult = self.strategy.get('keltner_mult', 2.0)
        df['kc_upper'], df['kc_basis'], df['kc_lower'] = self.calculate_keltner_channels(df, kc_length, kc_mult)

        # 3. Fisher Transform
        df['fisher'], df['fisher_signal'] = self._calculate_fisher_transform(df)

        # 4. FRAMA (Fractal Adaptive MA)
        df['frama'] = self._calculate_frama(df)

        # 5. Waddah Attar Explosion
        df['wae_trend'], df['wae_explosion'], df['wae_dead'] = self._calculate_waddah_attar(df)

        # 6. Squeeze Momentum (Bollinger + Keltner squeeze)
        df['squeeze_on'], df['squeeze_momentum'] = self._calculate_squeeze_momentum(df)

        # 7. Half Trend
        df['half_trend'], df['half_trend_dir'] = self._calculate_half_trend(df)

        # 8. SuperTrend
        st_length = self.strategy.get('supertrend_length', 22)
        st_mult = self.strategy.get('supertrend_mult', 3.0)
        df['supertrend'], df['supertrend_dir'] = self.calculate_supertrend(df, st_length, st_mult)

        # 9. RSI for confirmation
        df['rsi'] = self.calculate_rsi(df, length=14)

        # 10. Volatility Ratio
        df['volatility_ratio'] = self.calculate_volatility_ratio(df)

        # 11. Bollinger Bands for squeeze detection
        bb_length = self.strategy.get('bb_length', 20)
        bb_std = self.strategy.get('bb_std', 2.0)
        df['bb_upper'], df['bb_basis'], df['bb_lower'] = self.calculate_bollinger_bands(df, bb_length, bb_std)

        # 12. EMA trend
        df['ema_20'] = self.calculate_ema(df, 20)
        df['ema_50'] = self.calculate_ema(df, 50)
        df['ema_trend'] = np.where(df['ema_20'] > df['ema_50'], 1, -1)

        return df

    def _calculate_fisher_transform(self, df: pd.DataFrame, length: int = 9) -> Tuple[pd.Series, pd.Series]:
        """
        Fisher Transform - Converts prices to Gaussian normal distribution.
        Makes turning points more obvious.
        """
        try:
            high = df['high'].values
            low = df['low'].values

            hl2 = (high + low) / 2

            # Highest high and lowest low
            max_high = pd.Series(hl2).rolling(length).max().values
            min_low = pd.Series(hl2).rolling(length).min().values

            # Normalize to -1 to 1
            value = np.zeros(len(hl2))
            for i in range(length, len(hl2)):
                if max_high[i] != min_low[i]:
                    value[i] = 0.66 * ((hl2[i] - min_low[i]) / (max_high[i] - min_low[i]) - 0.5) + 0.67 * value[i-1]
                else:
                    value[i] = value[i-1]

            # Clamp to prevent infinity
            value = np.clip(value, -0.999, 0.999)

            # Fisher transform
            fisher = np.zeros(len(value))
            for i in range(1, len(value)):
                fisher[i] = 0.5 * np.log((1 + value[i]) / (1 - value[i])) + 0.5 * fisher[i-1]

            fisher_signal = pd.Series(fisher).shift(1).values

            return pd.Series(fisher, index=df.index), pd.Series(fisher_signal, index=df.index)

        except Exception as e:
            logger.error(f"Error calculating Fisher Transform: {e}")
            return pd.Series(0, index=df.index), pd.Series(0, index=df.index)

    def _calculate_frama(self, df: pd.DataFrame, length: int = 16) -> pd.Series:
        """Fractal Adaptive Moving Average"""
        try:
            close = df['close'].values
            frama = np.zeros(len(close))
            frama[:length] = close[:length]

            for i in range(length, len(close)):
                n = length // 2

                hl1 = np.max(close[i-length:i-n]) - np.min(close[i-length:i-n])
                hl2 = np.max(close[i-n:i]) - np.min(close[i-n:i])
                hl = np.max(close[i-length:i]) - np.min(close[i-length:i])

                if hl1 + hl2 == 0 or hl == 0:
                    alpha = 0.5
                else:
                    d = (np.log(hl1 + hl2) - np.log(hl)) / np.log(2)
                    alpha = np.exp(-4.6 * (d - 1))
                    alpha = np.clip(alpha, 0.01, 1.0)

                frama[i] = alpha * close[i] + (1 - alpha) * frama[i-1]

            return pd.Series(frama, index=df.index)

        except Exception as e:
            logger.error(f"Error calculating FRAMA: {e}")
            return pd.Series(df['close'].values, index=df.index)

    def _calculate_waddah_attar(self, df: pd.DataFrame,
                                 fast: int = 20, slow: int = 40,
                                 sensitivity: float = 150) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Waddah Attar Explosion - Entry timing indicator.
        Returns: (trend direction, explosion value, dead zone)
        """
        try:
            close = df['close'].values

            # MACD calculation
            fast_ma = pd.Series(close).ewm(span=fast, adjust=False).mean().values
            slow_ma = pd.Series(close).ewm(span=slow, adjust=False).mean().values
            macd = fast_ma - slow_ma

            # Trend direction
            trend = np.zeros(len(close))
            for i in range(1, len(close)):
                t1 = macd[i] - macd[i-1]
                trend[i] = 1 if t1 > 0 else -1

            # Explosion (strength)
            explosion = np.abs(macd) * sensitivity

            # Dead zone (BB width)
            bb = ta.bbands(pd.Series(close), length=20, std=2.0)
            bb_upper = bb[[c for c in bb.columns if 'BBU' in c][0]].values
            bb_lower = bb[[c for c in bb.columns if 'BBL' in c][0]].values
            dead_zone = (bb_upper - bb_lower) * sensitivity / 2

            return (pd.Series(trend, index=df.index),
                    pd.Series(explosion, index=df.index),
                    pd.Series(dead_zone, index=df.index))

        except Exception as e:
            logger.error(f"Error calculating Waddah Attar: {e}")
            return (pd.Series(0, index=df.index),
                    pd.Series(0, index=df.index),
                    pd.Series(0, index=df.index))

    def _calculate_squeeze_momentum(self, df: pd.DataFrame,
                                     bb_length: int = 20, bb_mult: float = 2.0,
                                     kc_length: int = 20, kc_mult: float = 1.5) -> Tuple[pd.Series, pd.Series]:
        """
        Squeeze Momentum Indicator.
        Squeeze ON when BB inside KC (low volatility).
        """
        try:
            # Bollinger Bands
            bb = ta.bbands(df['close'], length=bb_length, std=bb_mult)
            bb_upper = bb[[c for c in bb.columns if 'BBU' in c][0]]
            bb_lower = bb[[c for c in bb.columns if 'BBL' in c][0]]

            # Keltner Channels
            kc = ta.kc(df['high'], df['low'], df['close'], length=kc_length, scalar=kc_mult)
            kc_upper = kc[[c for c in kc.columns if 'KCU' in c][0]]
            kc_lower = kc[[c for c in kc.columns if 'KCL' in c][0]]

            # Squeeze: BB inside KC
            squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)

            # Momentum (linear regression of close)
            momentum = ta.linreg(df['close'], length=20)

            return squeeze_on, momentum

        except Exception as e:
            logger.error(f"Error calculating Squeeze Momentum: {e}")
            return pd.Series(False, index=df.index), pd.Series(0, index=df.index)

    def _calculate_half_trend(self, df: pd.DataFrame, amplitude: int = 2,
                               channel_deviation: float = 2.0) -> Tuple[pd.Series, pd.Series]:
        """
        Half Trend Indicator - Clean trend following signals.
        """
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values

            atr = ta.atr(df['high'], df['low'], df['close'], length=100).values

            trend = np.zeros(len(close))
            half_trend = np.zeros(len(close))

            for i in range(amplitude, len(close)):
                highest = np.max(high[i-amplitude:i+1])
                lowest = np.min(low[i-amplitude:i+1])

                # High/low average
                hl_avg = (highest + lowest) / 2

                if i == amplitude:
                    trend[i] = 1 if close[i] > hl_avg else -1
                    half_trend[i] = hl_avg
                else:
                    prev_trend = trend[i-1]

                    if prev_trend == 1:
                        max_low = np.max(low[i-amplitude:i+1])
                        if close[i] < max_low:
                            trend[i] = -1
                            half_trend[i] = highest
                        else:
                            trend[i] = 1
                            half_trend[i] = max(half_trend[i-1], max_low)
                    else:
                        min_high = np.min(high[i-amplitude:i+1])
                        if close[i] > min_high:
                            trend[i] = 1
                            half_trend[i] = lowest
                        else:
                            trend[i] = -1
                            half_trend[i] = min(half_trend[i-1], min_high)

            return pd.Series(half_trend, index=df.index), pd.Series(trend, index=df.index)

        except Exception as e:
            logger.error(f"Error calculating Half Trend: {e}")
            return pd.Series(df['close'].values, index=df.index), pd.Series(0, index=df.index)
