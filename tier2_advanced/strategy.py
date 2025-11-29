#!/usr/bin/env python3
"""
Tier 2 - Advanced Technical Strategy
Uses sophisticated technical indicators for signal generation.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Tuple
import sys
sys.path.insert(0, '../shared')

from base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class Tier2Strategy(BaseStrategy):
    """
    Tier 2 Advanced Technical Strategy

    Signal Logic:
    1. Squeeze Momentum - Volatility contraction + breakout
    2. Stochastic RSI - Precise momentum timing
    3. Fisher Transform - Clear turning points
    4. Keltner Channels - Volatility bands
    5. Waddah Attar - Entry confirmation
    6. Half Trend - Clean trend signals
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.strategy_name = "tier2_advanced"
        self.weights = self.get_weights()

    def get_weights(self) -> dict:
        """Get indicator weights for Tier 2."""
        return self.config.get('strategy', {}).get('weights', {
            'squeeze': 2.0,         # Squeeze breakout
            'stoch_rsi': 1.5,       # Momentum timing
            'fisher': 1.5,          # Turn detection
            'keltner': 1.0,         # Volatility bands
            'wae': 1.5,             # Entry timing
            'half_trend': 1.5,      # Trend direction
            'supertrend': 1.0,      # Trend confirmation
            'frama': 1.0            # Adaptive MA
        })

    def calculate_signal_score(self, action: str, last_bar: pd.Series,
                                prev_bar: pd.Series) -> Tuple[float, Dict]:
        """Calculate signal score for Tier 2 strategy."""
        score = 0
        indicators = {}
        weights = self.weights

        # 1. Squeeze Momentum
        squeeze_score, squeeze_data = self._score_squeeze(action, last_bar, prev_bar)
        score += squeeze_score * weights.get('squeeze', 2.0)
        indicators['squeeze'] = squeeze_data

        # 2. Stochastic RSI
        stoch_score, stoch_data = self._score_stoch_rsi(action, last_bar, prev_bar)
        score += stoch_score * weights.get('stoch_rsi', 1.5)
        indicators['stoch_rsi'] = stoch_data

        # 3. Fisher Transform
        fisher_score, fisher_data = self._score_fisher(action, last_bar, prev_bar)
        score += fisher_score * weights.get('fisher', 1.5)
        indicators['fisher'] = fisher_data

        # 4. Keltner Channels
        kc_score, kc_data = self._score_keltner(action, last_bar)
        score += kc_score * weights.get('keltner', 1.0)
        indicators['keltner'] = kc_data

        # 5. Waddah Attar Explosion
        wae_score, wae_data = self._score_waddah_attar(action, last_bar)
        score += wae_score * weights.get('wae', 1.5)
        indicators['wae'] = wae_data

        # 6. Half Trend
        ht_score, ht_data = self._score_half_trend(action, last_bar, prev_bar)
        score += ht_score * weights.get('half_trend', 1.5)
        indicators['half_trend'] = ht_data

        # 7. SuperTrend
        st_score, st_data = self._score_supertrend(action, last_bar, prev_bar)
        score += st_score * weights.get('supertrend', 1.0)
        indicators['supertrend'] = st_data

        # 8. FRAMA
        frama_score, frama_data = self._score_frama(action, last_bar)
        score += frama_score * weights.get('frama', 1.0)
        indicators['frama'] = frama_data

        return score, indicators

    def _score_squeeze(self, action: str, last_bar: pd.Series,
                       prev_bar: pd.Series) -> Tuple[float, Dict]:
        """Score based on Squeeze Momentum."""
        data = {'squeeze_on': False, 'momentum': 0, 'breakout': False}

        squeeze_on = last_bar.get('squeeze_on', False)
        squeeze_on_prev = prev_bar.get('squeeze_on', False)
        momentum = last_bar.get('squeeze_momentum', 0)
        momentum_prev = prev_bar.get('squeeze_momentum', 0)

        data['squeeze_on'] = squeeze_on
        data['momentum'] = momentum

        score = 0

        # Squeeze release (breakout)
        if squeeze_on_prev and not squeeze_on:
            data['breakout'] = True
            if action == 'LONG' and momentum > 0:
                score = 1.0
            elif action == 'SHORT' and momentum < 0:
                score = 1.0
        else:
            # During squeeze - look at momentum direction
            if action == 'LONG' and momentum > 0 and momentum > momentum_prev:
                score = 0.5
            elif action == 'SHORT' and momentum < 0 and momentum < momentum_prev:
                score = 0.5

        return score, data

    def _score_stoch_rsi(self, action: str, last_bar: pd.Series,
                         prev_bar: pd.Series) -> Tuple[float, Dict]:
        """Score based on Stochastic RSI."""
        data = {'k': 50, 'd': 50, 'cross': None}

        k = last_bar.get('stoch_rsi_k', 50)
        d = last_bar.get('stoch_rsi_d', 50)
        cross_up = last_bar.get('stoch_rsi_cross_up', False)
        cross_down = last_bar.get('stoch_rsi_cross_down', False)
        oversold = last_bar.get('stoch_rsi_oversold', False)
        overbought = last_bar.get('stoch_rsi_overbought', False)

        data['k'] = k
        data['d'] = d

        score = 0

        if action == 'LONG':
            if cross_up and oversold:
                data['cross'] = 'bullish_oversold'
                score = 1.0
            elif cross_up:
                data['cross'] = 'bullish'
                score = 0.7
            elif k > d and k < 80:
                score = 0.4

        else:  # SHORT
            if cross_down and overbought:
                data['cross'] = 'bearish_overbought'
                score = 1.0
            elif cross_down:
                data['cross'] = 'bearish'
                score = 0.7
            elif k < d and k > 20:
                score = 0.4

        return score, data

    def _score_fisher(self, action: str, last_bar: pd.Series,
                      prev_bar: pd.Series) -> Tuple[float, Dict]:
        """Score based on Fisher Transform."""
        data = {'fisher': 0, 'signal': 0, 'cross': None}

        fisher = last_bar.get('fisher', 0)
        signal = last_bar.get('fisher_signal', 0)
        fisher_prev = prev_bar.get('fisher', 0)
        signal_prev = prev_bar.get('fisher_signal', 0)

        data['fisher'] = fisher
        data['signal'] = signal

        score = 0

        # Fisher cross
        cross_up = fisher > signal and fisher_prev <= signal_prev
        cross_down = fisher < signal and fisher_prev >= signal_prev

        if action == 'LONG':
            if cross_up:
                data['cross'] = 'bullish'
                score = 0.8
            elif fisher > signal:
                score = 0.4

            # Extreme oversold turning up
            if fisher < -1.5 and fisher > fisher_prev:
                score = max(score, 0.9)

        else:  # SHORT
            if cross_down:
                data['cross'] = 'bearish'
                score = 0.8
            elif fisher < signal:
                score = 0.4

            if fisher > 1.5 and fisher < fisher_prev:
                score = max(score, 0.9)

        return score, data

    def _score_keltner(self, action: str, last_bar: pd.Series) -> Tuple[float, Dict]:
        """Score based on Keltner Channels."""
        data = {'position': 'middle'}

        close = last_bar['close']
        kc_upper = last_bar.get('kc_upper', close)
        kc_lower = last_bar.get('kc_lower', close)
        kc_basis = last_bar.get('kc_basis', close)

        score = 0

        if action == 'LONG':
            if close <= kc_lower:
                data['position'] = 'below_lower'
                score = 0.8  # Oversold
            elif close < kc_basis:
                data['position'] = 'below_basis'
                score = 0.5
            elif close > kc_upper:
                data['position'] = 'above_upper'
                score = 0.3  # Breakout but extended

        else:  # SHORT
            if close >= kc_upper:
                data['position'] = 'above_upper'
                score = 0.8  # Overbought
            elif close > kc_basis:
                data['position'] = 'above_basis'
                score = 0.5
            elif close < kc_lower:
                data['position'] = 'below_lower'
                score = 0.3

        return score, data

    def _score_waddah_attar(self, action: str, last_bar: pd.Series) -> Tuple[float, Dict]:
        """Score based on Waddah Attar Explosion."""
        data = {'trend': 0, 'explosion': 0, 'active': False}

        trend = last_bar.get('wae_trend', 0)
        explosion = last_bar.get('wae_explosion', 0)
        dead_zone = last_bar.get('wae_dead', 0)

        data['trend'] = trend
        data['explosion'] = explosion

        score = 0

        # Signal is active when explosion > dead zone
        active = explosion > dead_zone
        data['active'] = active

        if active:
            if action == 'LONG' and trend > 0:
                score = 1.0
            elif action == 'SHORT' and trend < 0:
                score = 1.0
        else:
            # Weak signal
            if action == 'LONG' and trend > 0:
                score = 0.3
            elif action == 'SHORT' and trend < 0:
                score = 0.3

        return score, data

    def _score_half_trend(self, action: str, last_bar: pd.Series,
                          prev_bar: pd.Series) -> Tuple[float, Dict]:
        """Score based on Half Trend."""
        data = {'direction': 0, 'flip': False}

        direction = last_bar.get('half_trend_dir', 0)
        direction_prev = prev_bar.get('half_trend_dir', 0)

        data['direction'] = direction

        score = 0

        # Check for flip
        flip_long = direction == 1 and direction_prev == -1
        flip_short = direction == -1 and direction_prev == 1

        if action == 'LONG':
            if flip_long:
                data['flip'] = True
                score = 1.0
            elif direction == 1:
                score = 0.6

        else:  # SHORT
            if flip_short:
                data['flip'] = True
                score = 1.0
            elif direction == -1:
                score = 0.6

        return score, data

    def _score_supertrend(self, action: str, last_bar: pd.Series,
                          prev_bar: pd.Series) -> Tuple[float, Dict]:
        """Score based on SuperTrend."""
        data = {'direction': 0}

        direction = last_bar.get('supertrend_dir', 0)
        data['direction'] = direction

        score = 0

        if action == 'LONG' and direction == 1:
            score = 0.7
        elif action == 'SHORT' and direction == -1:
            score = 0.7

        return score, data

    def _score_frama(self, action: str, last_bar: pd.Series) -> Tuple[float, Dict]:
        """Score based on FRAMA."""
        data = {'above': False}

        close = last_bar['close']
        frama = last_bar.get('frama', close)

        data['above'] = close > frama

        score = 0

        if action == 'LONG' and close > frama:
            score = 0.7
        elif action == 'SHORT' and close < frama:
            score = 0.7

        return score, data
