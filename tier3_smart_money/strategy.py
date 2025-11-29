#!/usr/bin/env python3
"""
Tier 3 - Smart Money Concepts Strategy
Uses ICT-style indicators for signal generation.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Tuple
import sys
sys.path.insert(0, '../shared')

from base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class Tier3Strategy(BaseStrategy):
    """
    Tier 3 Smart Money Strategy

    Signal Logic:
    1. Market Structure - BOS/CHoCH for direction
    2. Order Blocks - Entry zones
    3. Fair Value Gaps - Retracement targets
    4. Liquidity Grabs - Reversal signals
    5. Premium/Discount - Value positioning
    6. Confluence with EMA bias
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.strategy_name = "tier3_smart_money"
        self.weights = self.get_weights()

    def get_weights(self) -> dict:
        """Get indicator weights for Tier 3."""
        return self.config.get('strategy', {}).get('weights', {
            'structure': 2.5,        # BOS/CHoCH
            'order_block': 2.0,      # OB entry
            'fvg': 1.5,              # Fair value gap
            'liquidity': 2.0,        # Liquidity grab
            'premium_discount': 1.5, # Value zone
            'ema_bias': 1.0,         # Trend alignment
            'supertrend': 1.0        # Trend confirmation
        })

    def calculate_signal_score(self, action: str, last_bar: pd.Series,
                                prev_bar: pd.Series) -> Tuple[float, Dict]:
        """Calculate signal score for Tier 3 strategy."""
        score = 0
        indicators = {}
        weights = self.weights

        # 1. Market Structure (BOS/CHoCH)
        struct_score, struct_data = self._score_structure(action, last_bar)
        score += struct_score * weights.get('structure', 2.5)
        indicators['structure'] = struct_data

        # 2. Order Blocks
        ob_score, ob_data = self._score_order_blocks(action, last_bar)
        score += ob_score * weights.get('order_block', 2.0)
        indicators['order_block'] = ob_data

        # 3. Fair Value Gaps
        fvg_score, fvg_data = self._score_fvg(action, last_bar)
        score += fvg_score * weights.get('fvg', 1.5)
        indicators['fvg'] = fvg_data

        # 4. Liquidity Grabs
        liq_score, liq_data = self._score_liquidity(action, last_bar)
        score += liq_score * weights.get('liquidity', 2.0)
        indicators['liquidity'] = liq_data

        # 5. Premium/Discount Zones
        pd_score, pd_data = self._score_premium_discount(action, last_bar)
        score += pd_score * weights.get('premium_discount', 1.5)
        indicators['premium_discount'] = pd_data

        # 6. EMA Bias
        ema_score, ema_data = self._score_ema_bias(action, last_bar)
        score += ema_score * weights.get('ema_bias', 1.0)
        indicators['ema_bias'] = ema_data

        # 7. SuperTrend
        st_score, st_data = self._score_supertrend(action, last_bar)
        score += st_score * weights.get('supertrend', 1.0)
        indicators['supertrend'] = st_data

        return score, indicators

    def _score_structure(self, action: str, last_bar: pd.Series) -> Tuple[float, Dict]:
        """Score based on market structure (BOS/CHoCH)."""
        data = {'bos': False, 'choch': False, 'trend': 0}

        bos_bullish = last_bar.get('bos_bullish', 0)
        bos_bearish = last_bar.get('bos_bearish', 0)
        choch_bullish = last_bar.get('choch_bullish', 0)
        choch_bearish = last_bar.get('choch_bearish', 0)
        smc_trend = last_bar.get('smc_trend', 0)

        data['trend'] = smc_trend

        score = 0

        if action == 'LONG':
            if choch_bullish:
                data['choch'] = True
                score = 1.0  # CHoCH is strongest reversal signal
            elif bos_bullish:
                data['bos'] = True
                score = 0.8  # BOS is continuation
            elif smc_trend == 1:
                score = 0.4  # Aligned with trend

        else:  # SHORT
            if choch_bearish:
                data['choch'] = True
                score = 1.0
            elif bos_bearish:
                data['bos'] = True
                score = 0.8
            elif smc_trend == -1:
                score = 0.4

        return score, data

    def _score_order_blocks(self, action: str, last_bar: pd.Series) -> Tuple[float, Dict]:
        """Score based on Order Blocks."""
        data = {'at_ob': False, 'ob_type': None}

        at_bullish_ob = last_bar.get('at_bullish_ob', 0)
        at_bearish_ob = last_bar.get('at_bearish_ob', 0)

        score = 0

        if action == 'LONG':
            if at_bullish_ob:
                data['at_ob'] = True
                data['ob_type'] = 'bullish'
                score = 1.0  # At bullish order block

        else:  # SHORT
            if at_bearish_ob:
                data['at_ob'] = True
                data['ob_type'] = 'bearish'
                score = 1.0  # At bearish order block

        return score, data

    def _score_fvg(self, action: str, last_bar: pd.Series) -> Tuple[float, Dict]:
        """Score based on Fair Value Gaps."""
        data = {'at_fvg': False, 'fvg_type': None}

        at_bullish_fvg = last_bar.get('at_bullish_fvg', 0)
        at_bearish_fvg = last_bar.get('at_bearish_fvg', 0)

        score = 0

        if action == 'LONG':
            # Price filling a bullish FVG from above = potential support
            if at_bullish_fvg:
                data['at_fvg'] = True
                data['fvg_type'] = 'bullish'
                score = 0.8

        else:  # SHORT
            if at_bearish_fvg:
                data['at_fvg'] = True
                data['fvg_type'] = 'bearish'
                score = 0.8

        return score, data

    def _score_liquidity(self, action: str, last_bar: pd.Series) -> Tuple[float, Dict]:
        """Score based on Liquidity Grabs."""
        data = {'liq_grab': False, 'type': None}

        liq_grab_bullish = last_bar.get('liq_grab_bullish', 0)
        liq_grab_bearish = last_bar.get('liq_grab_bearish', 0)

        score = 0

        if action == 'LONG':
            if liq_grab_bullish:
                data['liq_grab'] = True
                data['type'] = 'bullish_reversal'
                score = 1.0  # Liquidity grab below = bullish reversal

        else:  # SHORT
            if liq_grab_bearish:
                data['liq_grab'] = True
                data['type'] = 'bearish_reversal'
                score = 1.0  # Liquidity grab above = bearish reversal

        return score, data

    def _score_premium_discount(self, action: str, last_bar: pd.Series) -> Tuple[float, Dict]:
        """Score based on Premium/Discount zones."""
        data = {'zone': 'equilibrium'}

        in_premium = last_bar.get('in_premium', 0)
        in_discount = last_bar.get('in_discount', 0)

        score = 0

        if action == 'LONG':
            if in_discount:
                data['zone'] = 'discount'
                score = 1.0  # Buy in discount zone
            elif in_premium:
                data['zone'] = 'premium'
                score = 0.2  # Buying in premium is risky

        else:  # SHORT
            if in_premium:
                data['zone'] = 'premium'
                score = 1.0  # Sell in premium zone
            elif in_discount:
                data['zone'] = 'discount'
                score = 0.2  # Selling in discount is risky

        return score, data

    def _score_ema_bias(self, action: str, last_bar: pd.Series) -> Tuple[float, Dict]:
        """Score based on EMA bias."""
        data = {'bias': 0}

        ema_bias = last_bar.get('ema_bias', 0)
        data['bias'] = ema_bias

        score = 0

        if action == 'LONG' and ema_bias == 1:
            score = 0.7
        elif action == 'SHORT' and ema_bias == -1:
            score = 0.7

        return score, data

    def _score_supertrend(self, action: str, last_bar: pd.Series) -> Tuple[float, Dict]:
        """Score based on SuperTrend."""
        data = {'direction': 0}

        direction = last_bar.get('supertrend_dir', 0)
        data['direction'] = direction

        score = 0

        if action == 'LONG' and direction == 1:
            score = 0.6
        elif action == 'SHORT' and direction == -1:
            score = 0.6

        return score, data

    def _create_signal(self, symbol: str, action: str, last_bar: pd.Series,
                       score: float, indicators: dict, bar_time) -> Dict:
        """Create signal with SMC-specific stops."""
        signal = super()._create_signal(symbol, action, last_bar, score, indicators, bar_time)

        # Use Order Block levels for stops if available
        if action == 'LONG':
            ob_bottom = last_bar.get('ob_bottom', 0)
            if ob_bottom > 0:
                signal['stop_loss'] = ob_bottom * 0.998  # Just below OB
        else:
            ob_top = last_bar.get('ob_top', 0)
            if ob_top > 0:
                signal['stop_loss'] = ob_top * 1.002  # Just above OB

        # Recalculate TP based on new SL
        entry = signal['entry_price']
        sl = signal['stop_loss']
        risk = abs(entry - sl)
        if action == 'LONG':
            signal['take_profit'] = entry + (risk * self.risk_reward_ratio)
        else:
            signal['take_profit'] = entry - (risk * self.risk_reward_ratio)

        return signal
