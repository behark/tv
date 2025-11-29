#!/usr/bin/env python3
"""
Enhanced Accuracy Filters for Multi-Tier Trading Bots
Provides quality filtering, scoring, and signal validation.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas_ta as ta

logger = logging.getLogger(__name__)


class SignalScorer:
    """
    Scores trading signals based on multiple factors.
    Higher scores indicate higher confidence signals.
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.weights = self.config.get('scoring_weights', {
            'trend_alignment': 2.0,
            'momentum': 1.5,
            'volume': 1.0,
            'volatility': 1.0,
            'mtf_confirmation': 2.5,
            'support_resistance': 1.5,
            'funding_sentiment': 1.0,
            'order_flow': 1.5
        })
        self.min_score = self.config.get('min_signal_score', 7.0)
        self.max_score = sum(self.weights.values())

    def calculate_score(self, signal: Dict, df: pd.DataFrame,
                        htf_trend: str = None,
                        funding_rate: float = None,
                        order_imbalance: float = None) -> Tuple[float, Dict]:
        """
        Calculate comprehensive signal score.
        Returns: Tuple of (score, breakdown_dict)
        """
        breakdown = {}
        total_score = 0

        action = signal.get('action', 'LONG')
        last_bar = df.iloc[-1]

        # 1. Trend Alignment Score
        trend_score = self._score_trend_alignment(action, last_bar, df)
        breakdown['trend_alignment'] = trend_score
        total_score += trend_score * self.weights.get('trend_alignment', 2.0)

        # 2. Momentum Score
        momentum_score = self._score_momentum(action, last_bar)
        breakdown['momentum'] = momentum_score
        total_score += momentum_score * self.weights.get('momentum', 1.5)

        # 3. Volume Score
        volume_score = self._score_volume(last_bar)
        breakdown['volume'] = volume_score
        total_score += volume_score * self.weights.get('volume', 1.0)

        # 4. Volatility Score
        volatility_score = self._score_volatility(last_bar, df)
        breakdown['volatility'] = volatility_score
        total_score += volatility_score * self.weights.get('volatility', 1.0)

        # 5. Multi-Timeframe Confirmation
        mtf_score = self._score_mtf_confirmation(action, htf_trend)
        breakdown['mtf_confirmation'] = mtf_score
        total_score += mtf_score * self.weights.get('mtf_confirmation', 2.5)

        # 6. Support/Resistance Proximity
        sr_score = self._score_support_resistance(action, last_bar, df)
        breakdown['support_resistance'] = sr_score
        total_score += sr_score * self.weights.get('support_resistance', 1.5)

        # 7. Funding Rate Sentiment (Crypto specific)
        if funding_rate is not None:
            funding_score = self._score_funding_sentiment(action, funding_rate)
            breakdown['funding_sentiment'] = funding_score
            total_score += funding_score * self.weights.get('funding_sentiment', 1.0)

        # 8. Order Flow Imbalance
        if order_imbalance is not None:
            flow_score = self._score_order_flow(action, order_imbalance)
            breakdown['order_flow'] = flow_score
            total_score += flow_score * self.weights.get('order_flow', 1.5)

        # Normalize to 10-point scale
        normalized_score = (total_score / self.max_score) * 10

        logger.debug(f"Signal score: {normalized_score:.1f}/10 - {breakdown}")

        return normalized_score, breakdown

    def _score_trend_alignment(self, action: str, last_bar: pd.Series,
                                df: pd.DataFrame) -> float:
        """Score based on trend alignment"""
        score = 0
        close = last_bar['close']

        # Check various trend indicators
        trend_indicators = ['frama', 'ema', 'vwap', 'ichimoku_base']
        for ind in trend_indicators:
            if ind in last_bar:
                if action == 'LONG' and close > last_bar[ind]:
                    score += 0.25
                elif action == 'SHORT' and close < last_bar[ind]:
                    score += 0.25

        # Check SuperTrend direction
        if 'supertrend_dir' in last_bar:
            if action == 'LONG' and last_bar['supertrend_dir'] == 1:
                score += 0.25
            elif action == 'SHORT' and last_bar['supertrend_dir'] == -1:
                score += 0.25

        return min(score, 1.0)

    def _score_momentum(self, action: str, last_bar: pd.Series) -> float:
        """Score based on momentum indicators"""
        score = 0

        # RSX/RSI score
        for rsi_col in ['rsx', 'rsi', 'stoch_rsi']:
            if rsi_col in last_bar:
                rsi_val = last_bar[rsi_col]
                if action == 'LONG':
                    if 30 <= rsi_val <= 50:
                        score += 0.3
                    elif 50 < rsi_val <= 65:
                        score += 0.15
                else:
                    if 50 <= rsi_val <= 70:
                        score += 0.3
                    elif 35 <= rsi_val < 50:
                        score += 0.15
                break

        # MACD alignment
        if 'macd' in last_bar and 'macd_signal' in last_bar:
            if action == 'LONG' and last_bar['macd'] > last_bar['macd_signal']:
                score += 0.2
            elif action == 'SHORT' and last_bar['macd'] < last_bar['macd_signal']:
                score += 0.2

        # ADX strength
        if 'adx' in last_bar:
            adx = last_bar['adx']
            if adx >= 25:
                score += 0.3
            elif adx >= 20:
                score += 0.15

        return min(score, 1.0)

    def _score_volume(self, last_bar: pd.Series) -> float:
        """Score based on volume confirmation"""
        if 'volume' not in last_bar or 'avg_volume' not in last_bar:
            return 0.5

        volume = last_bar['volume']
        avg_volume = last_bar['avg_volume']

        if avg_volume == 0:
            return 0.5

        ratio = volume / avg_volume

        if ratio >= 2.0:
            return 1.0
        elif ratio >= 1.5:
            return 0.8
        elif ratio >= 1.2:
            return 0.6
        elif ratio >= 1.0:
            return 0.4
        else:
            return 0.2

    def _score_volatility(self, last_bar: pd.Series, df: pd.DataFrame) -> float:
        """Score based on volatility conditions"""
        score = 0.5

        if 'volatility_ratio' in last_bar:
            vr = last_bar['volatility_ratio']
            if 0.3 <= vr <= 0.7:
                score = 0.8
            elif vr < 0.3:
                score = 0.4
            else:
                score = 0.6

        if 'atr' in last_bar:
            atr_pct = (last_bar['atr'] / last_bar['close']) * 100
            if 0.5 <= atr_pct <= 3.0:
                score = max(score, 0.7)

        return score

    def _score_mtf_confirmation(self, action: str, htf_trend: str) -> float:
        """Score based on higher timeframe confirmation"""
        if htf_trend is None:
            return 0.5

        if action == htf_trend:
            return 1.0
        elif htf_trend == 'NEUTRAL':
            return 0.5
        else:
            return 0.0

    def _score_support_resistance(self, action: str, last_bar: pd.Series,
                                   df: pd.DataFrame) -> float:
        """Score based on proximity to support/resistance levels"""
        score = 0.5
        close = last_bar['close']

        # Check band indicators
        for band_prefix in ['iqr', 'bb', 'kc']:
            upper_col = f'{band_prefix}_upper'
            lower_col = f'{band_prefix}_lower'

            if upper_col in last_bar and lower_col in last_bar:
                upper = last_bar[upper_col]
                lower = last_bar[lower_col]

                if action == 'LONG':
                    distance_to_support = (close - lower) / close * 100
                    if distance_to_support < 1.0:
                        score = 0.9
                    elif distance_to_support < 2.0:
                        score = 0.7
                else:
                    distance_to_resistance = (upper - close) / close * 100
                    if distance_to_resistance < 1.0:
                        score = 0.9
                    elif distance_to_resistance < 2.0:
                        score = 0.7
                break

        return score

    def _score_funding_sentiment(self, action: str, funding_rate: float) -> float:
        """
        Score based on funding rate sentiment.
        Positive funding = longs pay shorts (bullish crowd)
        Negative funding = shorts pay longs (bearish crowd)
        Contrarian: go against extreme funding
        """
        if funding_rate is None:
            return 0.5

        # Extreme funding thresholds
        extreme_positive = 0.01  # 0.01% per 8h = very bullish crowd
        extreme_negative = -0.01

        if action == 'LONG':
            if funding_rate < extreme_negative:
                return 1.0  # Contrarian long when crowd is very bearish
            elif funding_rate < 0:
                return 0.7
            elif funding_rate > extreme_positive:
                return 0.2  # Avoid longs when everyone is long
            else:
                return 0.5
        else:  # SHORT
            if funding_rate > extreme_positive:
                return 1.0  # Contrarian short when crowd is very bullish
            elif funding_rate > 0:
                return 0.7
            elif funding_rate < extreme_negative:
                return 0.2
            else:
                return 0.5

    def _score_order_flow(self, action: str, imbalance: float) -> float:
        """
        Score based on order book imbalance.
        Imbalance > 0 = more bids than asks (bullish)
        Imbalance < 0 = more asks than bids (bearish)
        """
        if imbalance is None:
            return 0.5

        if action == 'LONG':
            if imbalance > 0.3:
                return 1.0  # Strong bid support
            elif imbalance > 0.1:
                return 0.7
            elif imbalance < -0.3:
                return 0.2  # Heavy selling pressure
            else:
                return 0.5
        else:  # SHORT
            if imbalance < -0.3:
                return 1.0  # Strong sell pressure
            elif imbalance < -0.1:
                return 0.7
            elif imbalance > 0.3:
                return 0.2
            else:
                return 0.5

    def passes_threshold(self, score: float) -> bool:
        """Check if score passes minimum threshold"""
        return score >= self.min_score


class MultiTimeframeAnalyzer:
    """Analyzes higher timeframes for trend confirmation."""

    TIMEFRAME_HIERARCHY = {
        '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
        '1h': 60, '2h': 120, '4h': 240, '6h': 360,
        '12h': 720, '1d': 1440, '1w': 10080
    }

    def __init__(self, data_client, config: dict = None):
        self.data_client = data_client
        self.config = config or {}
        self._htf_cache = {}
        self._cache_ttl = 300

    def get_higher_timeframe(self, current_tf: str) -> str:
        """Get the next higher timeframe"""
        htf_mapping = {
            '1m': '5m', '3m': '15m', '5m': '15m',
            '15m': '1h', '30m': '1h', '1h': '4h',
            '2h': '4h', '4h': '1d', '6h': '1d',
            '12h': '1d', '1d': '1w'
        }
        return htf_mapping.get(current_tf, '4h')

    def analyze_htf_trend(self, symbol: str, htf: str = '1h') -> Dict:
        """Analyze higher timeframe trend."""
        cache_key = f"{symbol}_{htf}"
        now = datetime.now()

        if cache_key in self._htf_cache:
            cached, cached_time = self._htf_cache[cache_key]
            if (now - cached_time).total_seconds() < self._cache_ttl:
                return cached

        try:
            df = self.data_client.fetch_ohlcv(symbol, htf, limit=100)

            if df is None or len(df) < 50:
                return {'trend': 'NEUTRAL', 'confidence': 0}

            df['ema20'] = ta.ema(df['close'], length=20)
            df['ema50'] = ta.ema(df['close'], length=50)
            df['rsi'] = ta.rsi(df['close'], length=14)

            last = df.iloc[-1]

            trend = 'NEUTRAL'
            confidence = 0

            if last['close'] > last['ema20'] > last['ema50']:
                trend = 'LONG'
                confidence += 40
            elif last['close'] < last['ema20'] < last['ema50']:
                trend = 'SHORT'
                confidence += 40

            if len(df) >= 5:
                recent_highs = df['high'].tail(5)
                recent_lows = df['low'].tail(5)

                if recent_highs.is_monotonic_increasing and recent_lows.is_monotonic_increasing:
                    if trend == 'LONG':
                        confidence += 30
                    trend = 'LONG'
                elif recent_highs.is_monotonic_decreasing and recent_lows.is_monotonic_decreasing:
                    if trend == 'SHORT':
                        confidence += 30
                    trend = 'SHORT'

            rsi = last['rsi']
            if trend == 'LONG' and 40 <= rsi <= 70:
                confidence += 20
            elif trend == 'SHORT' and 30 <= rsi <= 60:
                confidence += 20

            confidence = min(confidence, 100)

            result = {
                'trend': trend,
                'confidence': confidence,
                'ema20': last['ema20'],
                'ema50': last['ema50'],
                'rsi': rsi,
                'timeframe': htf
            }

            self._htf_cache[cache_key] = (result, now)
            return result

        except Exception as e:
            logger.error(f"Error analyzing HTF for {symbol}: {e}")
            return {'trend': 'NEUTRAL', 'confidence': 0}

    def confirms_signal(self, signal: Dict, htf_analysis: Dict) -> bool:
        """Check if HTF analysis confirms the signal"""
        if htf_analysis['trend'] == 'NEUTRAL':
            return True

        min_confidence = self.config.get('mtf_min_confidence', 50)

        if htf_analysis['confidence'] < min_confidence:
            return True

        return signal['action'] == htf_analysis['trend']


class VolumeFilter:
    """Filter signals based on volume conditions"""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.min_volume_ratio = self.config.get('min_volume_ratio', 1.0)
        self.spike_threshold = self.config.get('volume_spike_threshold', 2.0)

    def check_volume(self, df: pd.DataFrame) -> Dict:
        """Check volume conditions."""
        last = df.iloc[-1]

        if 'volume' not in last or 'avg_volume' not in last:
            return {'passes': True, 'ratio': 1.0, 'is_spike': False}

        volume = last['volume']
        avg_volume = last['avg_volume']

        if avg_volume == 0:
            return {'passes': True, 'ratio': 1.0, 'is_spike': False}

        ratio = volume / avg_volume
        is_spike = ratio >= self.spike_threshold
        passes = ratio >= self.min_volume_ratio

        return {
            'passes': passes,
            'ratio': ratio,
            'is_spike': is_spike,
            'volume': volume,
            'avg_volume': avg_volume
        }


class VolatilityFilter:
    """Filter signals based on volatility conditions"""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.max_atr_percent = self.config.get('max_atr_percent', 5.0)
        self.min_atr_percent = self.config.get('min_atr_percent', 0.3)

    def check_volatility(self, df: pd.DataFrame) -> Dict:
        """Check volatility conditions."""
        last = df.iloc[-1]

        if 'atr' not in last:
            return {'passes': True, 'atr_percent': 1.0, 'condition': 'normal'}

        atr = last['atr']
        close = last['close']
        atr_percent = (atr / close) * 100

        if atr_percent > self.max_atr_percent:
            condition = 'extreme'
            passes = False
        elif atr_percent < self.min_atr_percent:
            condition = 'low'
            passes = False
        else:
            condition = 'normal'
            passes = True

        return {
            'passes': passes,
            'atr_percent': atr_percent,
            'atr': atr,
            'condition': condition
        }


class AccuracyFilterManager:
    """Manages all accuracy filters and provides unified filtering interface."""

    def __init__(self, data_client, config: dict = None):
        self.config = config or {}
        self.data_client = data_client

        self.scorer = SignalScorer(config)
        self.mtf_analyzer = MultiTimeframeAnalyzer(data_client, config)
        self.volume_filter = VolumeFilter(config)
        self.volatility_filter = VolatilityFilter(config)

        self.enable_mtf = self.config.get('enable_mtf_filter', True)
        self.enable_volume = self.config.get('enable_volume_filter', True)
        self.enable_volatility = self.config.get('enable_volatility_filter', True)
        self.enable_scoring = self.config.get('enable_signal_scoring', True)

        logger.info(f"Accuracy filters initialized")

    def filter_signal(self, signal: Dict, df: pd.DataFrame,
                      timeframe: str = '15m',
                      funding_rate: float = None,
                      order_imbalance: float = None) -> Tuple[bool, Dict]:
        """Apply all filters to a signal."""
        results = {
            'passes': True,
            'score': None,
            'score_breakdown': None,
            'mtf': None,
            'volume': None,
            'volatility': None,
            'rejection_reason': None
        }

        symbol = signal['symbol']

        # 1. Multi-timeframe confirmation
        if self.enable_mtf:
            htf = self.mtf_analyzer.get_higher_timeframe(timeframe)
            htf_analysis = self.mtf_analyzer.analyze_htf_trend(symbol, htf)
            results['mtf'] = htf_analysis

            if not self.mtf_analyzer.confirms_signal(signal, htf_analysis):
                results['passes'] = False
                results['rejection_reason'] = f"HTF trend ({htf_analysis['trend']}) conflicts"
                return False, results

        # 2. Volume filter
        if self.enable_volume:
            volume_check = self.volume_filter.check_volume(df)
            results['volume'] = volume_check

            if not volume_check['passes']:
                results['passes'] = False
                results['rejection_reason'] = f"Low volume ({volume_check['ratio']:.2f}x)"
                return False, results

        # 3. Volatility filter
        if self.enable_volatility:
            volatility_check = self.volatility_filter.check_volatility(df)
            results['volatility'] = volatility_check

            if not volatility_check['passes']:
                results['passes'] = False
                results['rejection_reason'] = f"Bad volatility ({volatility_check['condition']})"
                return False, results

        # 4. Signal scoring
        if self.enable_scoring:
            htf_trend = results['mtf']['trend'] if results['mtf'] else None
            score, breakdown = self.scorer.calculate_score(
                signal, df, htf_trend, funding_rate, order_imbalance
            )
            results['score'] = score
            results['score_breakdown'] = breakdown

            if not self.scorer.passes_threshold(score):
                results['passes'] = False
                results['rejection_reason'] = f"Low score ({score:.1f}/10)"
                return False, results

        return True, results

    def get_enhanced_signal(self, signal: Dict, filter_results: Dict) -> Dict:
        """Enhance signal with filter data"""
        enhanced = signal.copy()

        if filter_results.get('score'):
            enhanced['quality_score'] = filter_results['score']
            enhanced['score_breakdown'] = filter_results.get('score_breakdown', {})

        if filter_results.get('mtf'):
            enhanced['htf_trend'] = filter_results['mtf']['trend']
            enhanced['htf_confidence'] = filter_results['mtf']['confidence']

        if filter_results.get('volume'):
            enhanced['volume_ratio'] = filter_results['volume'].get('ratio', 1.0)
            enhanced['volume_spike'] = filter_results['volume'].get('is_spike', False)

        return enhanced
