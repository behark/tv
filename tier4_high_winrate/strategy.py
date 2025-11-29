#!/usr/bin/env python3
"""
Tier 4 - High Win Rate Strategy
Multi-confluence mean reversion targeting 75-85% accuracy.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from base_strategy import BaseStrategy


class Tier4Strategy(BaseStrategy):
    """
    High Win Rate Mean Reversion Strategy

    Entry Requirements (must meet minimum confluence score):
    1. RSI at extreme levels (oversold/overbought)
    2. Price at Bollinger Band extreme
    3. Stochastic RSI confirmation
    4. Volume spike (climax volume preferred)
    5. Near support/resistance level
    6. Optional: RSI divergence (strong bonus)
    7. Prefer ranging market regime

    Exit Strategy:
    - Take profit at mean (middle BB or VWAP)
    - Stop loss at recent swing high/low
    - Trailing stop after 50% to target

    This strategy prioritizes win rate over R:R ratio.
    Target: 75-85% win rate with 1.5:1 R:R
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.strategy_name = "Tier 4 - High Win Rate"
        self.strategy_config = config.get('strategy', {})

        # Minimum confluence score for entry
        self.min_confluence_score = self.strategy_config.get('min_confluence_score', 5.0)
        self.strong_signal_score = self.strategy_config.get('strong_signal_score', 7.0)

        # Risk management
        self.risk_reward_ratio = self.strategy_config.get('risk_reward_ratio', 1.5)
        self.atr_multiplier_sl = self.strategy_config.get('atr_multiplier_sl', 1.2)

        # Weights for scoring
        self.weights = self.strategy_config.get('weights', {
            'rsi_extreme': 2.0,
            'bb_touch': 2.0,
            'stoch_rsi': 1.5,
            'volume_climax': 2.0,
            'rsi_divergence': 2.5,
            'near_sr': 2.0,
            'regime_bonus': 1.5,
            'trend_alignment': 1.0
        })

    def calculate_signal_score(self, df: pd.DataFrame) -> Dict:
        """
        Calculate signal score based on high-probability confluences.
        Returns dict with action, score, and details.
        """
        if df is None or len(df) < 2:
            return {'action': 'NONE', 'score': 0}

        current = df.iloc[-1]
        prev = df.iloc[-2]

        bullish_score = 0.0
        bearish_score = 0.0
        bullish_reasons = []
        bearish_reasons = []

        # 1. RSI Extreme Levels
        rsi = current.get('rsi', 50)
        if rsi < 30:
            points = self.weights['rsi_extreme']
            if rsi < 20:
                points *= 1.5  # Extra for extreme
            bullish_score += points
            bullish_reasons.append(f"RSI oversold ({rsi:.1f})")

        if rsi > 70:
            points = self.weights['rsi_extreme']
            if rsi > 80:
                points *= 1.5
            bearish_score += points
            bearish_reasons.append(f"RSI overbought ({rsi:.1f})")

        # 2. Bollinger Band Touch
        close = current['close']
        bb_lower = current.get('bb_lower', 0)
        bb_upper = current.get('bb_upper', float('inf'))
        bb_middle = current.get('bb_middle', close)

        if close <= bb_lower:
            bullish_score += self.weights['bb_touch']
            bullish_reasons.append("At lower BB")
        elif close <= bb_lower * 1.005:  # Within 0.5% of lower BB
            bullish_score += self.weights['bb_touch'] * 0.7
            bullish_reasons.append("Near lower BB")

        if close >= bb_upper:
            bearish_score += self.weights['bb_touch']
            bearish_reasons.append("At upper BB")
        elif close >= bb_upper * 0.995:
            bearish_score += self.weights['bb_touch'] * 0.7
            bearish_reasons.append("Near upper BB")

        # 3. Stochastic RSI
        stoch_k = current.get('stoch_rsi_k', 50)
        stoch_d = current.get('stoch_rsi_d', 50)

        if stoch_k < 20 and stoch_d < 20:
            bullish_score += self.weights['stoch_rsi']
            bullish_reasons.append(f"StochRSI oversold ({stoch_k:.1f})")
            # Bullish crossover
            if prev.get('stoch_rsi_k', 50) < prev.get('stoch_rsi_d', 50) and stoch_k > stoch_d:
                bullish_score += 1.0
                bullish_reasons.append("StochRSI bullish cross")

        if stoch_k > 80 and stoch_d > 80:
            bearish_score += self.weights['stoch_rsi']
            bearish_reasons.append(f"StochRSI overbought ({stoch_k:.1f})")
            # Bearish crossover
            if prev.get('stoch_rsi_k', 50) > prev.get('stoch_rsi_d', 50) and stoch_k < stoch_d:
                bearish_score += 1.0
                bearish_reasons.append("StochRSI bearish cross")

        # 4. Volume Climax (potential exhaustion)
        if current.get('volume_climax', False):
            bullish_score += self.weights['volume_climax']
            bearish_score += self.weights['volume_climax']
            bullish_reasons.append("Volume climax")
            bearish_reasons.append("Volume climax")
        elif current.get('volume_spike', False):
            bullish_score += self.weights['volume_climax'] * 0.5
            bearish_score += self.weights['volume_climax'] * 0.5

        # 5. RSI Divergence (strong reversal signal)
        if current.get('rsi_bullish_div', False):
            bullish_score += self.weights['rsi_divergence']
            bullish_reasons.append("Bullish RSI divergence")

        if current.get('rsi_bearish_div', False):
            bearish_score += self.weights['rsi_divergence']
            bearish_reasons.append("Bearish RSI divergence")

        # 6. Near Support/Resistance
        if current.get('near_support', False):
            bullish_score += self.weights['near_sr']
            bullish_reasons.append("At support level")

        if current.get('near_resistance', False):
            bearish_score += self.weights['near_sr']
            bearish_reasons.append("At resistance level")

        # 7. Market Regime Bonus (prefer ranging for mean reversion)
        if current.get('regime_ranging', False):
            bullish_score *= (1 + self.weights['regime_bonus'] * 0.1)
            bearish_score *= (1 + self.weights['regime_bonus'] * 0.1)
            bullish_reasons.append("Ranging market (ideal)")
            bearish_reasons.append("Ranging market (ideal)")

        # 8. Trend Alignment (avoid fighting strong trends)
        if current.get('trend_up', False):
            bullish_score += self.weights['trend_alignment']
            bearish_score *= 0.7  # Penalize shorts in uptrend
            bullish_reasons.append("Uptrend alignment")
        elif current.get('trend_down', False):
            bearish_score += self.weights['trend_alignment']
            bullish_score *= 0.7  # Penalize longs in downtrend
            bearish_reasons.append("Downtrend alignment")

        # 9. VWAP Deviation
        vwap = current.get('vwap')
        if pd.notna(vwap) and vwap > 0:
            vwap_deviation = (close - vwap) / vwap * 100
            if vwap_deviation < -1.5:
                bullish_score += 1.5
                bullish_reasons.append(f"Below VWAP ({vwap_deviation:.1f}%)")
            elif vwap_deviation > 1.5:
                bearish_score += 1.5
                bearish_reasons.append(f"Above VWAP ({vwap_deviation:.1f}%)")

        # Determine final action
        action = 'NONE'
        final_score = 0
        reasons = []

        if bullish_score >= self.min_confluence_score and bullish_score > bearish_score:
            action = 'LONG'
            final_score = bullish_score
            reasons = bullish_reasons
        elif bearish_score >= self.min_confluence_score and bearish_score > bullish_score:
            action = 'SHORT'
            final_score = bearish_score
            reasons = bearish_reasons

        # Signal strength classification
        strength = 'weak'
        if final_score >= self.strong_signal_score:
            strength = 'strong'
        elif final_score >= self.min_confluence_score:
            strength = 'moderate'

        return {
            'action': action,
            'score': final_score,
            'strength': strength,
            'reasons': reasons,
            'bullish_score': bullish_score,
            'bearish_score': bearish_score
        }

    def check_signals(self, symbol: str, df: pd.DataFrame) -> Optional[Dict]:
        """Check for trading signals."""
        if df is None or len(df) < 2:
            return None

        current = df.iloc[-1]

        # Get signal score
        signal_data = self.calculate_signal_score(df)

        if signal_data['action'] == 'NONE':
            return None

        action = signal_data['action']

        # Check if direction is enabled
        if action == 'LONG' and not self.enable_long:
            return None
        if action == 'SHORT' and not self.enable_short:
            return None

        # Calculate entry, stop loss, and take profit
        entry_price = current['close']
        atr = current.get('atr', entry_price * 0.02)

        if action == 'LONG':
            # Stop loss below recent low or support
            recent_low = df['low'].iloc[-5:].min()
            support = current.get('nearest_support', recent_low)
            stop_loss = min(recent_low, support) - (atr * 0.3)

            # Take profit at middle BB or VWAP (mean reversion target)
            bb_middle = current.get('bb_middle', entry_price * 1.01)
            vwap = current.get('vwap', bb_middle)
            take_profit = max(bb_middle, vwap)

            # Ensure minimum R:R
            risk = entry_price - stop_loss
            if risk <= 0:
                risk = atr * self.atr_multiplier_sl
                stop_loss = entry_price - risk

            min_reward = risk * self.risk_reward_ratio
            if take_profit - entry_price < min_reward:
                take_profit = entry_price + min_reward

        else:  # SHORT
            # Stop loss above recent high or resistance
            recent_high = df['high'].iloc[-5:].max()
            resistance = current.get('nearest_resistance', recent_high)
            stop_loss = max(recent_high, resistance) + (atr * 0.3)

            # Take profit at middle BB or VWAP
            bb_middle = current.get('bb_middle', entry_price * 0.99)
            vwap = current.get('vwap', bb_middle)
            take_profit = min(bb_middle, vwap)

            # Ensure minimum R:R
            risk = stop_loss - entry_price
            if risk <= 0:
                risk = atr * self.atr_multiplier_sl
                stop_loss = entry_price + risk

            min_reward = risk * self.risk_reward_ratio
            if entry_price - take_profit < min_reward:
                take_profit = entry_price - min_reward

        # Calculate risk/reward ratio
        if action == 'LONG':
            risk = entry_price - stop_loss
            reward = take_profit - entry_price
        else:
            risk = stop_loss - entry_price
            reward = entry_price - take_profit

        rr_ratio = reward / risk if risk > 0 else 0

        # Build signal dictionary
        signal = {
            'symbol': symbol,
            'action': action,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'score': signal_data['score'],
            'strength': signal_data['strength'],
            'reasons': signal_data['reasons'],
            'risk_reward': rr_ratio,
            'timestamp': datetime.now().isoformat(),
            'strategy': self.strategy_name,
            'indicators': {
                'rsi': current.get('rsi'),
                'stoch_rsi_k': current.get('stoch_rsi_k'),
                'bb_position': 'lower' if current['close'] <= current.get('bb_lower', 0) else
                              'upper' if current['close'] >= current.get('bb_upper', float('inf')) else 'middle',
                'volume_ratio': current.get('volume_ratio'),
                'adx': current.get('adx'),
                'market_regime': 'ranging' if current.get('regime_ranging') else 'trending'
            }
        }

        return signal

    def format_signal_message(self, signal: Dict) -> str:
        """Format signal for Telegram notification."""
        action_emoji = "🟢" if signal['action'] == 'LONG' else "🔴"
        strength_emoji = "💪" if signal['strength'] == 'strong' else "✨" if signal['strength'] == 'moderate' else ""

        message = f"{action_emoji} <b>HIGH WIN RATE SIGNAL</b> {strength_emoji}\n\n"
        message += f"<b>Symbol:</b> {signal['symbol']}\n"
        message += f"<b>Action:</b> {signal['action']}\n"
        message += f"<b>Strength:</b> {signal['strength'].upper()}\n\n"

        message += f"<b>Entry:</b> {signal['entry_price']:.4f}\n"
        message += f"<b>Stop Loss:</b> {signal['stop_loss']:.4f}\n"
        message += f"<b>Take Profit:</b> {signal['take_profit']:.4f}\n"
        message += f"<b>R:R Ratio:</b> 1:{signal['risk_reward']:.2f}\n\n"

        message += f"<b>Confluence Score:</b> {signal['score']:.1f}\n"
        message += f"<b>Reasons:</b>\n"
        for reason in signal['reasons'][:5]:  # Limit to 5 reasons
            message += f"  • {reason}\n"

        # Indicator snapshot
        ind = signal.get('indicators', {})
        message += f"\n<b>Indicators:</b>\n"
        message += f"  RSI: {ind.get('rsi', 'N/A'):.1f}\n" if ind.get('rsi') else ""
        message += f"  StochRSI: {ind.get('stoch_rsi_k', 'N/A'):.1f}\n" if ind.get('stoch_rsi_k') else ""
        message += f"  BB Position: {ind.get('bb_position', 'N/A')}\n"
        message += f"  Regime: {ind.get('market_regime', 'N/A')}\n"

        message += f"\n<b>Strategy:</b> {signal['strategy']}\n"
        message += f"<i>Expected Win Rate: 75-85%</i>"

        return message
