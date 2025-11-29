#!/usr/bin/env python3
"""
Tier 1 - Institutional Strategy
Uses institutional-grade indicators for signal generation.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Tuple
import sys
sys.path.insert(0, '../shared')

from base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class Tier1Strategy(BaseStrategy):
    """
    Tier 1 Institutional Strategy

    Signal Logic:
    1. VWAP Position - Price relative to VWAP determines bias
    2. Ichimoku Cloud - Trend confirmation
    3. CVD Direction - Volume delta confirms direction
    4. EMA Alignment - Trend strength
    5. Volume Confirmation - CMF/OBV confirmation
    6. Funding Rate - Contrarian positioning on extremes
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.strategy_name = "tier1_institutional"
        self.weights = self.get_weights()

    def get_weights(self) -> dict:
        """Get indicator weights for Tier 1."""
        return self.config.get('strategy', {}).get('weights', {
            'vwap': 2.0,           # VWAP is key institutional level
            'ichimoku': 2.0,       # Full trend system
            'supertrend': 1.5,     # Trend direction
            'cvd': 1.5,            # Volume delta
            'ema_alignment': 1.0,  # Trend strength
            'volume_flow': 1.0,    # CMF/OBV
            'momentum': 1.0,       # RSI/MFI
            'funding': 0.5         # Contrarian signal
        })

    def calculate_signal_score(self, action: str, last_bar: pd.Series,
                                prev_bar: pd.Series) -> Tuple[float, Dict]:
        """Calculate signal score for Tier 1 strategy."""
        score = 0
        indicators = {}
        weights = self.weights

        # 1. VWAP Position (key institutional level)
        vwap_score, vwap_data = self._score_vwap(action, last_bar)
        score += vwap_score * weights.get('vwap', 2.0)
        indicators['vwap'] = vwap_data

        # 2. Ichimoku Cloud
        ichimoku_score, ichimoku_data = self._score_ichimoku(action, last_bar, prev_bar)
        score += ichimoku_score * weights.get('ichimoku', 2.0)
        indicators['ichimoku'] = ichimoku_data

        # 3. SuperTrend
        st_score, st_data = self._score_supertrend(action, last_bar, prev_bar)
        score += st_score * weights.get('supertrend', 1.5)
        indicators['supertrend'] = st_data

        # 4. CVD (Cumulative Volume Delta)
        cvd_score, cvd_data = self._score_cvd(action, last_bar)
        score += cvd_score * weights.get('cvd', 1.5)
        indicators['cvd'] = cvd_data

        # 5. EMA Alignment
        ema_score, ema_data = self._score_ema_alignment(action, last_bar)
        score += ema_score * weights.get('ema_alignment', 1.0)
        indicators['ema_alignment'] = ema_data

        # 6. Volume Flow (CMF + OBV)
        vol_score, vol_data = self._score_volume_flow(action, last_bar)
        score += vol_score * weights.get('volume_flow', 1.0)
        indicators['volume_flow'] = vol_data

        # 7. Momentum (RSI + MFI)
        mom_score, mom_data = self._score_momentum(action, last_bar, prev_bar)
        score += mom_score * weights.get('momentum', 1.0)
        indicators['momentum'] = mom_data

        return score, indicators

    def _score_vwap(self, action: str, last_bar: pd.Series) -> Tuple[float, Dict]:
        """Score based on VWAP position."""
        data = {'position': 'neutral', 'distance': 0}

        if 'vwap' not in last_bar:
            return 0.5, data

        close = last_bar['close']
        vwap = last_bar['vwap']
        vwap_distance = last_bar.get('vwap_distance', 0)

        data['vwap'] = vwap
        data['distance'] = vwap_distance

        score = 0

        if action == 'LONG':
            # Best: Price above VWAP but not too extended
            if close > vwap:
                data['position'] = 'above'
                if 0 < vwap_distance < 1.0:
                    score = 1.0  # Ideal: just above VWAP
                elif vwap_distance < 2.0:
                    score = 0.7  # Still good
                else:
                    score = 0.3  # Extended
            else:
                # Price below VWAP - look for bounce
                data['position'] = 'below'
                if 'vwap_lower' in last_bar and close <= last_bar['vwap_lower']:
                    score = 0.8  # At lower band - potential reversal
                else:
                    score = 0.4

        else:  # SHORT
            if close < vwap:
                data['position'] = 'below'
                if -1.0 < vwap_distance < 0:
                    score = 1.0
                elif vwap_distance > -2.0:
                    score = 0.7
                else:
                    score = 0.3
            else:
                data['position'] = 'above'
                if 'vwap_upper' in last_bar and close >= last_bar['vwap_upper']:
                    score = 0.8
                else:
                    score = 0.4

        return score, data

    def _score_ichimoku(self, action: str, last_bar: pd.Series,
                        prev_bar: pd.Series) -> Tuple[float, Dict]:
        """Score based on Ichimoku Cloud."""
        data = {'cloud': 'neutral', 'tk_cross': False}
        score = 0

        above_cloud = last_bar.get('above_cloud', False)
        below_cloud = last_bar.get('below_cloud', False)
        cloud_bullish = last_bar.get('ichimoku_cloud_bullish', True)

        # Tenkan/Kijun cross
        tenkan = last_bar.get('ichimoku_tenkan', 0)
        kijun = last_bar.get('ichimoku_kijun', 0)
        tenkan_prev = prev_bar.get('ichimoku_tenkan', 0)
        kijun_prev = prev_bar.get('ichimoku_kijun', 0)

        tk_cross_up = tenkan > kijun and tenkan_prev <= kijun_prev
        tk_cross_down = tenkan < kijun and tenkan_prev >= kijun_prev

        if action == 'LONG':
            if above_cloud:
                data['cloud'] = 'above'
                score += 0.4
            if cloud_bullish:
                score += 0.2
            if tenkan > kijun:
                score += 0.2
            if tk_cross_up:
                data['tk_cross'] = True
                score += 0.2

        else:  # SHORT
            if below_cloud:
                data['cloud'] = 'below'
                score += 0.4
            if not cloud_bullish:
                score += 0.2
            if tenkan < kijun:
                score += 0.2
            if tk_cross_down:
                data['tk_cross'] = True
                score += 0.2

        return min(score, 1.0), data

    def _score_supertrend(self, action: str, last_bar: pd.Series,
                          prev_bar: pd.Series) -> Tuple[float, Dict]:
        """Score based on SuperTrend."""
        data = {'direction': 0, 'flip': False}

        direction = last_bar.get('supertrend_dir', 0)
        direction_prev = prev_bar.get('supertrend_dir', 0)

        data['direction'] = direction

        # Check for flip
        flip_long = direction == 1 and direction_prev == -1
        flip_short = direction == -1 and direction_prev == 1

        score = 0

        if action == 'LONG':
            if direction == 1:
                score = 0.7
            if flip_long:
                data['flip'] = True
                score = 1.0

        else:  # SHORT
            if direction == -1:
                score = 0.7
            if flip_short:
                data['flip'] = True
                score = 1.0

        return score, data

    def _score_cvd(self, action: str, last_bar: pd.Series) -> Tuple[float, Dict]:
        """Score based on CVD (Cumulative Volume Delta)."""
        data = {'signal': 0, 'trend': 'neutral'}

        cvd_signal = last_bar.get('cvd_signal', 0)
        data['signal'] = cvd_signal

        score = 0

        if action == 'LONG':
            if cvd_signal == 1:
                data['trend'] = 'bullish'
                score = 1.0
            else:
                score = 0.3

        else:  # SHORT
            if cvd_signal == -1:
                data['trend'] = 'bearish'
                score = 1.0
            else:
                score = 0.3

        return score, data

    def _score_ema_alignment(self, action: str, last_bar: pd.Series) -> Tuple[float, Dict]:
        """Score based on EMA alignment."""
        data = {'alignment': 0}

        alignment = last_bar.get('ema_alignment', 0)
        data['alignment'] = alignment

        # Alignment ranges from -4 to +4 (5 EMAs)
        max_alignment = 4

        if action == 'LONG':
            if alignment > 0:
                score = alignment / max_alignment
            else:
                score = 0.1
        else:
            if alignment < 0:
                score = abs(alignment) / max_alignment
            else:
                score = 0.1

        return min(score, 1.0), data

    def _score_volume_flow(self, action: str, last_bar: pd.Series) -> Tuple[float, Dict]:
        """Score based on CMF and OBV."""
        data = {'cmf': 0, 'obv_signal': 0}

        cmf = last_bar.get('cmf', 0)
        obv_signal = last_bar.get('obv_signal', 0)

        data['cmf'] = cmf
        data['obv_signal'] = obv_signal

        score = 0

        if action == 'LONG':
            if cmf > 0:
                score += 0.5
            if obv_signal == 1:
                score += 0.5
        else:
            if cmf < 0:
                score += 0.5
            if obv_signal == -1:
                score += 0.5

        return score, data

    def _score_momentum(self, action: str, last_bar: pd.Series,
                        prev_bar: pd.Series) -> Tuple[float, Dict]:
        """Score based on RSI and MFI."""
        data = {'rsi': 50, 'mfi': 50, 'divergence': None}

        rsi = last_bar.get('rsi', 50)
        mfi = last_bar.get('mfi', 50)
        bullish_div = last_bar.get('rsi_bullish_div', False)
        bearish_div = last_bar.get('rsi_bearish_div', False)

        data['rsi'] = rsi
        data['mfi'] = mfi

        score = 0

        if action == 'LONG':
            # Ideal RSI zone for longs: 30-60
            if 30 <= rsi <= 60:
                score += 0.4
            elif rsi < 30:  # Oversold
                score += 0.5

            # MFI confirmation
            if mfi < 80:
                score += 0.3

            # Bullish divergence is strong signal
            if bullish_div:
                data['divergence'] = 'bullish'
                score += 0.3

        else:  # SHORT
            if 40 <= rsi <= 70:
                score += 0.4
            elif rsi > 70:  # Overbought
                score += 0.5

            if mfi > 20:
                score += 0.3

            if bearish_div:
                data['divergence'] = 'bearish'
                score += 0.3

        return min(score, 1.0), data


class Tier1SignalGenerator:
    """
    Complete signal generator for Tier 1.
    Combines strategy with institutional data analysis.
    """

    def __init__(self, config: dict, data_enhancer=None):
        self.config = config
        self.strategy = Tier1Strategy(config)
        self.data_enhancer = data_enhancer

    def generate_signal(self, symbol: str, df: pd.DataFrame) -> Optional[Dict]:
        """Generate a signal with institutional data enhancement."""

        # Get base signal from strategy
        signal = self.strategy.check_signals(symbol, df)

        if signal is None:
            return None

        # Enhance with institutional data if available
        if self.data_enhancer:
            sentiment = self.data_enhancer.analyze_institutional_sentiment(symbol)

            signal['funding_rate'] = sentiment.get('funding_rate')
            signal['funding_sentiment'] = sentiment.get('funding_sentiment')
            signal['open_interest'] = sentiment.get('open_interest')
            signal['order_imbalance'] = sentiment.get('order_imbalance')
            signal['institutional_sentiment'] = sentiment.get('overall_sentiment')

            # Adjust score based on funding (contrarian)
            if signal['funding_rate'] is not None:
                rate = signal['funding_rate']
                action = signal['action']

                # Contrarian: if everyone is long (positive funding), favor shorts
                if action == 'LONG' and rate > 0.001:
                    signal['score'] *= 0.9  # Slight penalty
                elif action == 'LONG' and rate < -0.001:
                    signal['score'] *= 1.1  # Bonus for contrarian

                if action == 'SHORT' and rate < -0.001:
                    signal['score'] *= 0.9
                elif action == 'SHORT' and rate > 0.001:
                    signal['score'] *= 1.1

        return signal
