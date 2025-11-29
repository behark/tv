#!/usr/bin/env python3
"""
Tier 3 - Smart Money Concepts Indicators
Focus: Order Blocks, Fair Value Gaps, Liquidity Zones, Market Structure

These indicators are based on ICT (Inner Circle Trader) concepts and
attempt to identify where "smart money" is operating.
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
import logging
from typing import Dict, Tuple, Optional, List
import sys
sys.path.insert(0, '../shared')

from base_indicators import BaseIndicatorCalculator

logger = logging.getLogger(__name__)


class Tier3Indicators(BaseIndicatorCalculator):
    """
    Tier 3 Smart Money Indicators:
    1. Order Blocks - Institutional entry zones
    2. Fair Value Gaps (FVG) - Imbalance zones
    3. Liquidity Zones - Stop hunt levels
    4. Market Structure - Higher highs/lows
    5. Break of Structure (BOS) - Trend changes
    6. Change of Character (CHoCH) - Reversal signals
    7. Premium/Discount Zones - Value areas
    8. Swing Points - Key levels
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.strategy = config.get('strategy', {})

    def calculate_tier_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Tier 3 specific indicators."""

        # 1. Market Structure (Swing highs/lows)
        swing_lookback = self.strategy.get('swing_lookback', 5)
        structure = self.calculate_market_structure(df, swing_lookback)
        df['swing_high'] = structure['swing_high']
        df['swing_low'] = structure['swing_low']
        df['structure'] = structure['structure']

        # 2. Order Blocks
        df = self._calculate_order_blocks(df)

        # 3. Fair Value Gaps (FVG)
        df = self._calculate_fvg(df)

        # 4. Liquidity Zones
        df = self._calculate_liquidity_zones(df)

        # 5. Break of Structure (BOS) / Change of Character (CHoCH)
        df = self._calculate_bos_choch(df)

        # 6. Premium/Discount Zones
        df = self._calculate_premium_discount(df)

        # 7. RSI for confluence
        df['rsi'] = self.calculate_rsi(df, length=14)

        # 8. SuperTrend for trend
        st_length = self.strategy.get('supertrend_length', 22)
        st_mult = self.strategy.get('supertrend_mult', 3.0)
        df['supertrend'], df['supertrend_dir'] = self.calculate_supertrend(df, st_length, st_mult)

        # 9. EMA for bias
        df['ema_50'] = self.calculate_ema(df, 50)
        df['ema_200'] = self.calculate_ema(df, 200)
        df['ema_bias'] = np.where(df['ema_50'] > df['ema_200'], 1, -1)

        # 10. Volatility Ratio
        df['volatility_ratio'] = self.calculate_volatility_ratio(df)

        return df

    def _calculate_order_blocks(self, df: pd.DataFrame,
                                 lookback: int = 10) -> pd.DataFrame:
        """
        Identify Order Blocks - Last bullish/bearish candle before a strong move.
        Bullish OB: Last bearish candle before a strong bullish move
        Bearish OB: Last bullish candle before a strong bearish move
        """
        high = df['high'].values
        low = df['low'].values
        open_p = df['open'].values
        close = df['close'].values

        bullish_ob = np.zeros(len(df))
        bearish_ob = np.zeros(len(df))
        ob_top = np.zeros(len(df))
        ob_bottom = np.zeros(len(df))

        # Minimum move to qualify as "strong" (in ATR)
        atr = df['atr'].values if 'atr' in df.columns else np.ones(len(df))
        min_move = 1.5  # 1.5x ATR

        for i in range(lookback, len(df)):
            # Look for bearish candle followed by strong bullish move (Bullish OB)
            if close[i-1] < open_p[i-1]:  # Previous candle is bearish
                # Check if current candle is strongly bullish
                move = close[i] - low[i-1]
                if move > min_move * atr[i]:
                    bullish_ob[i] = 1
                    ob_top[i] = high[i-1]
                    ob_bottom[i] = low[i-1]

            # Look for bullish candle followed by strong bearish move (Bearish OB)
            if close[i-1] > open_p[i-1]:  # Previous candle is bullish
                move = high[i-1] - close[i]
                if move > min_move * atr[i]:
                    bearish_ob[i] = 1
                    ob_top[i] = high[i-1]
                    ob_bottom[i] = low[i-1]

        df['bullish_ob'] = bullish_ob
        df['bearish_ob'] = bearish_ob
        df['ob_top'] = ob_top
        df['ob_bottom'] = ob_bottom

        # Track if price is at an order block
        df['at_bullish_ob'] = self._price_at_level(df, 'bullish_ob', 'ob_bottom', 'ob_top', lookback=20)
        df['at_bearish_ob'] = self._price_at_level(df, 'bearish_ob', 'ob_bottom', 'ob_top', lookback=20)

        return df

    def _calculate_fvg(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Fair Value Gaps (imbalance zones).
        Bullish FVG: Gap up - Low of current > High of 2 bars ago
        Bearish FVG: Gap down - High of current < Low of 2 bars ago
        """
        high = df['high'].values
        low = df['low'].values

        bullish_fvg = np.zeros(len(df))
        bearish_fvg = np.zeros(len(df))
        fvg_top = np.zeros(len(df))
        fvg_bottom = np.zeros(len(df))

        for i in range(2, len(df)):
            # Bullish FVG: gap up
            if low[i] > high[i-2]:
                bullish_fvg[i] = 1
                fvg_top[i] = low[i]
                fvg_bottom[i] = high[i-2]

            # Bearish FVG: gap down
            if high[i] < low[i-2]:
                bearish_fvg[i] = 1
                fvg_top[i] = low[i-2]
                fvg_bottom[i] = high[i]

        df['bullish_fvg'] = bullish_fvg
        df['bearish_fvg'] = bearish_fvg
        df['fvg_top'] = fvg_top
        df['fvg_bottom'] = fvg_bottom

        # Track if price is filling an FVG
        df['at_bullish_fvg'] = self._price_at_level(df, 'bullish_fvg', 'fvg_bottom', 'fvg_top', lookback=20)
        df['at_bearish_fvg'] = self._price_at_level(df, 'bearish_fvg', 'fvg_bottom', 'fvg_top', lookback=20)

        return df

    def _calculate_liquidity_zones(self, df: pd.DataFrame,
                                    lookback: int = 20) -> pd.DataFrame:
        """
        Identify liquidity zones where stops are likely clustered.
        - Above swing highs (buy stops)
        - Below swing lows (sell stops)
        """
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        # Recent swing highs/lows
        swing_high = df['swing_high'].values if 'swing_high' in df.columns else np.zeros(len(df))
        swing_low = df['swing_low'].values if 'swing_low' in df.columns else np.zeros(len(df))

        liquidity_above = np.zeros(len(df))
        liquidity_below = np.zeros(len(df))
        liq_grab_above = np.zeros(len(df))
        liq_grab_below = np.zeros(len(df))

        for i in range(lookback, len(df)):
            # Find recent swing highs (liquidity above)
            recent_swing_highs = [swing_high[j] for j in range(i-lookback, i) if swing_high[j] > 0]
            if recent_swing_highs:
                liquidity_above[i] = max(recent_swing_highs)

                # Check for liquidity grab (price sweeps above then closes below)
                if high[i] > liquidity_above[i] and close[i] < liquidity_above[i]:
                    liq_grab_above[i] = 1

            # Find recent swing lows (liquidity below)
            recent_swing_lows = [swing_low[j] for j in range(i-lookback, i) if swing_low[j] > 0]
            if recent_swing_lows:
                liquidity_below[i] = min(recent_swing_lows)

                # Check for liquidity grab (price sweeps below then closes above)
                if low[i] < liquidity_below[i] and close[i] > liquidity_below[i]:
                    liq_grab_below[i] = 1

        df['liquidity_above'] = liquidity_above
        df['liquidity_below'] = liquidity_below
        df['liq_grab_bullish'] = liq_grab_below  # Grab below = bullish reversal
        df['liq_grab_bearish'] = liq_grab_above  # Grab above = bearish reversal

        return df

    def _calculate_bos_choch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Break of Structure (BOS) and Change of Character (CHoCH).
        BOS: Price breaks a swing point in trend direction (continuation)
        CHoCH: Price breaks a swing point against trend (reversal)
        """
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        swing_high = df['swing_high'].values if 'swing_high' in df.columns else np.zeros(len(df))
        swing_low = df['swing_low'].values if 'swing_low' in df.columns else np.zeros(len(df))

        bos_bullish = np.zeros(len(df))
        bos_bearish = np.zeros(len(df))
        choch_bullish = np.zeros(len(df))
        choch_bearish = np.zeros(len(df))

        trend = 0  # 1 = bullish, -1 = bearish
        last_swing_high = 0
        last_swing_low = float('inf')

        for i in range(len(df)):
            # Update swing points
            if swing_high[i] > 0:
                if swing_high[i] > last_swing_high and trend == 1:
                    pass  # Higher high in uptrend
                elif swing_high[i] < last_swing_high and trend == 1:
                    pass  # Lower high in uptrend - potential CHoCH
                last_swing_high = swing_high[i]

            if swing_low[i] > 0:
                last_swing_low = swing_low[i]

            # BOS Bullish: Break above last swing high in uptrend
            if trend >= 0 and close[i] > last_swing_high and last_swing_high > 0:
                bos_bullish[i] = 1
                trend = 1

            # BOS Bearish: Break below last swing low in downtrend
            if trend <= 0 and close[i] < last_swing_low and last_swing_low < float('inf'):
                bos_bearish[i] = 1
                trend = -1

            # CHoCH Bullish: Break above swing high in downtrend
            if trend == -1 and close[i] > last_swing_high and last_swing_high > 0:
                choch_bullish[i] = 1
                trend = 1

            # CHoCH Bearish: Break below swing low in uptrend
            if trend == 1 and close[i] < last_swing_low and last_swing_low < float('inf'):
                choch_bearish[i] = 1
                trend = -1

        df['bos_bullish'] = bos_bullish
        df['bos_bearish'] = bos_bearish
        df['choch_bullish'] = choch_bullish
        df['choch_bearish'] = choch_bearish
        df['smc_trend'] = pd.Series([1 if bos_bullish[i] or choch_bullish[i] else
                                      (-1 if bos_bearish[i] or choch_bearish[i] else 0)
                                      for i in range(len(df))]).replace(0, np.nan).ffill().fillna(0)

        return df

    def _calculate_premium_discount(self, df: pd.DataFrame,
                                     lookback: int = 50) -> pd.DataFrame:
        """
        Calculate Premium and Discount zones.
        Premium: Upper 50% of range (expensive to buy)
        Discount: Lower 50% of range (cheap to buy)
        """
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        premium = np.zeros(len(df))
        discount = np.zeros(len(df))
        equilibrium = np.zeros(len(df))

        for i in range(lookback, len(df)):
            range_high = np.max(high[i-lookback:i+1])
            range_low = np.min(low[i-lookback:i+1])
            range_mid = (range_high + range_low) / 2

            equilibrium[i] = range_mid

            if close[i] > range_mid:
                premium[i] = 1
                discount[i] = 0
            else:
                premium[i] = 0
                discount[i] = 1

        df['in_premium'] = premium
        df['in_discount'] = discount
        df['equilibrium'] = equilibrium

        return df

    def _price_at_level(self, df: pd.DataFrame, signal_col: str,
                        bottom_col: str, top_col: str,
                        lookback: int = 20) -> pd.Series:
        """Check if current price is at a previously identified level."""
        close = df['close'].values
        result = np.zeros(len(df))

        for i in range(lookback, len(df)):
            for j in range(i-lookback, i):
                if df[signal_col].iloc[j] == 1:
                    bottom = df[bottom_col].iloc[j]
                    top = df[top_col].iloc[j]
                    if bottom <= close[i] <= top:
                        result[i] = 1
                        break

        return pd.Series(result, index=df.index)
