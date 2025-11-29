#!/usr/bin/env python3
"""
Tier 1 - Institutional Indicators
Focus: VWAP, CVD, Funding Rate, Open Interest, Ichimoku

These are indicators commonly used by institutional traders and provide
insight into "smart money" positioning.
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


class Tier1Indicators(BaseIndicatorCalculator):
    """
    Tier 1 Institutional Indicators:
    1. VWAP - Volume Weighted Average Price (institutional benchmark)
    2. CVD - Cumulative Volume Delta (buying/selling pressure)
    3. Ichimoku Cloud - Multi-component trend system
    4. SuperTrend - Trend direction with ATR-based stops
    5. EMA Ribbon - Multiple EMAs for trend strength
    6. OBV - On Balance Volume
    7. CMF - Chaikin Money Flow
    8. RSI with Divergence detection
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.strategy = config.get('strategy', {})

    def calculate_tier_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Tier 1 specific indicators."""

        # 1. VWAP - The institutional price benchmark
        df['vwap'] = self.calculate_vwap(df)

        # VWAP bands (standard deviation bands around VWAP)
        df = self._calculate_vwap_bands(df)

        # 2. CVD - Cumulative Volume Delta
        df['cvd'] = self.calculate_cvd(df)
        df['cvd_ema'] = ta.ema(df['cvd'], length=20)
        df['cvd_signal'] = np.where(df['cvd'] > df['cvd_ema'], 1, -1)

        # 3. Ichimoku Cloud
        ichimoku = self.calculate_ichimoku(df)
        for key, value in ichimoku.items():
            df[f'ichimoku_{key}'] = value

        # Cloud color (bullish/bearish)
        if 'ichimoku_senkou_a' in df.columns and 'ichimoku_senkou_b' in df.columns:
            df['ichimoku_cloud_bullish'] = df['ichimoku_senkou_a'] > df['ichimoku_senkou_b']

        # Price vs Cloud
        df['above_cloud'] = (df['close'] > df['ichimoku_senkou_a']) & (df['close'] > df['ichimoku_senkou_b'])
        df['below_cloud'] = (df['close'] < df['ichimoku_senkou_a']) & (df['close'] < df['ichimoku_senkou_b'])

        # 4. SuperTrend
        st_length = self.strategy.get('supertrend_length', 22)
        st_mult = self.strategy.get('supertrend_mult', 3.0)
        df['supertrend'], df['supertrend_dir'] = self.calculate_supertrend(df, st_length, st_mult)

        # 5. EMA Ribbon (multiple EMAs for trend strength)
        ema_periods = [8, 13, 21, 34, 55]
        for period in ema_periods:
            df[f'ema_{period}'] = self.calculate_ema(df, period)

        # EMA alignment score
        df['ema_alignment'] = self._calculate_ema_alignment(df, ema_periods)

        # 6. OBV - On Balance Volume
        df['obv'] = self.calculate_obv(df)
        df['obv_ema'] = ta.ema(df['obv'], length=20)
        df['obv_signal'] = np.where(df['obv'] > df['obv_ema'], 1, -1)

        # 7. CMF - Chaikin Money Flow
        df['cmf'] = self.calculate_cmf(df, length=20)

        # 8. RSI
        df['rsi'] = self.calculate_rsi(df, length=14)

        # RSI Divergence
        df = self._detect_rsi_divergence(df)

        # 9. MFI - Money Flow Index (volume-weighted RSI)
        df['mfi'] = self.calculate_mfi(df, length=14)

        # 10. Volatility Ratio
        df['volatility_ratio'] = self.calculate_volatility_ratio(df)

        # 11. Price position relative to VWAP
        df['vwap_distance'] = (df['close'] - df['vwap']) / df['vwap'] * 100

        return df

    def _calculate_vwap_bands(self, df: pd.DataFrame, std_mult: float = 2.0) -> pd.DataFrame:
        """Calculate VWAP with standard deviation bands."""
        # Calculate cumulative values for the session
        cumulative_tp_vol = ((df['high'] + df['low'] + df['close']) / 3 * df['volume']).cumsum()
        cumulative_vol = df['volume'].cumsum()

        vwap = cumulative_tp_vol / cumulative_vol

        # Standard deviation around VWAP
        tp = (df['high'] + df['low'] + df['close']) / 3
        squared_diff = ((tp - vwap) ** 2 * df['volume']).cumsum()
        variance = squared_diff / cumulative_vol
        std = np.sqrt(variance)

        df['vwap_upper'] = vwap + std_mult * std
        df['vwap_lower'] = vwap - std_mult * std

        return df

    def _calculate_ema_alignment(self, df: pd.DataFrame, periods: list) -> pd.Series:
        """
        Calculate EMA alignment score.
        +1 for each EMA in bullish order, -1 for bearish.
        """
        alignment = pd.Series(0, index=df.index)

        for i in range(len(periods) - 1):
            faster = f'ema_{periods[i]}'
            slower = f'ema_{periods[i+1]}'

            if faster in df.columns and slower in df.columns:
                alignment += np.where(df[faster] > df[slower], 1, -1)

        return alignment

    def _detect_rsi_divergence(self, df: pd.DataFrame, lookback: int = 14) -> pd.DataFrame:
        """
        Detect RSI divergence (bullish and bearish).
        Bullish: Price makes lower low, RSI makes higher low
        Bearish: Price makes higher high, RSI makes lower high
        """
        df['rsi_bullish_div'] = False
        df['rsi_bearish_div'] = False

        price = df['close'].values
        rsi = df['rsi'].values

        for i in range(lookback, len(df)):
            # Look for swing points in the lookback window
            price_window = price[i-lookback:i+1]
            rsi_window = rsi[i-lookback:i+1]

            # Find local minima/maxima
            price_min_idx = np.argmin(price_window)
            price_max_idx = np.argmax(price_window)

            # Bullish divergence: price lower low, RSI higher low
            if price_min_idx == lookback:  # Current bar is lowest
                prev_min_idx = np.argmin(price_window[:-1])
                if (price_window[-1] < price_window[prev_min_idx] and
                    rsi_window[-1] > rsi_window[prev_min_idx]):
                    df.iloc[i, df.columns.get_loc('rsi_bullish_div')] = True

            # Bearish divergence: price higher high, RSI lower high
            if price_max_idx == lookback:  # Current bar is highest
                prev_max_idx = np.argmax(price_window[:-1])
                if (price_window[-1] > price_window[prev_max_idx] and
                    rsi_window[-1] < rsi_window[prev_max_idx]):
                    df.iloc[i, df.columns.get_loc('rsi_bearish_div')] = True

        return df


class Tier1DataEnhancer:
    """
    Enhances OHLCV data with institutional data:
    - Funding rates
    - Open Interest
    - Order book imbalance
    """

    def __init__(self, data_client):
        self.data_client = data_client
        self._funding_cache = {}
        self._oi_cache = {}

    def get_funding_rate(self, symbol: str) -> Optional[Dict]:
        """Get current funding rate for a symbol."""
        return self.data_client.fetch_funding_rate(symbol)

    def get_open_interest(self, symbol: str) -> Optional[Dict]:
        """Get current open interest for a symbol."""
        return self.data_client.fetch_open_interest(symbol)

    def get_order_imbalance(self, symbol: str) -> Optional[float]:
        """Get order book imbalance."""
        ob = self.data_client.fetch_order_book(symbol)
        if ob:
            return ob.get('imbalance', 0)
        return None

    def analyze_institutional_sentiment(self, symbol: str) -> Dict:
        """
        Analyze institutional sentiment from multiple sources.
        Returns sentiment score and components.
        """
        result = {
            'funding_rate': None,
            'funding_sentiment': 'neutral',
            'open_interest': None,
            'oi_change': None,
            'order_imbalance': None,
            'overall_sentiment': 'neutral',
            'sentiment_score': 0
        }

        # Funding rate analysis
        funding = self.get_funding_rate(symbol)
        if funding:
            result['funding_rate'] = funding.get('rate', 0)
            rate = result['funding_rate']

            if rate > 0.0005:  # Very positive = crowded long
                result['funding_sentiment'] = 'crowded_long'
                result['sentiment_score'] -= 1
            elif rate < -0.0005:  # Very negative = crowded short
                result['funding_sentiment'] = 'crowded_short'
                result['sentiment_score'] += 1
            elif rate > 0:
                result['funding_sentiment'] = 'slightly_bullish'
            else:
                result['funding_sentiment'] = 'slightly_bearish'

        # Open interest
        oi = self.get_open_interest(symbol)
        if oi:
            result['open_interest'] = oi.get('open_interest', 0)

        # Order book imbalance
        imbalance = self.get_order_imbalance(symbol)
        if imbalance is not None:
            result['order_imbalance'] = imbalance

            if imbalance > 0.2:
                result['sentiment_score'] += 1
            elif imbalance < -0.2:
                result['sentiment_score'] -= 1

        # Overall sentiment
        if result['sentiment_score'] > 0:
            result['overall_sentiment'] = 'bullish'
        elif result['sentiment_score'] < 0:
            result['overall_sentiment'] = 'bearish'

        return result
