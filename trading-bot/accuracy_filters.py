#!/usr/bin/env python3
"""
Accuracy Filters Module
Implements multi-timeframe confirmation, signal scoring, and various filters
to improve signal quality and reduce false positives.
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
            'support_resistance': 1.5
        })
        self.min_score = self.config.get('min_signal_score', 7.0)
        self.max_score = sum(self.weights.values())

    def calculate_score(self, signal: Dict, df: pd.DataFrame,
                        htf_trend: str = None) -> Tuple[float, Dict]:
        """
        Calculate comprehensive signal score.

        Returns:
            Tuple of (score, breakdown_dict)
        """
        breakdown = {}
        total_score = 0

        action = signal.get('action', 'LONG')
        last_bar = df.iloc[-1]

        # 1. Trend Alignment Score
        trend_score = self._score_trend_alignment(action, last_bar, df)
        breakdown['trend_alignment'] = trend_score
        total_score += trend_score * self.weights['trend_alignment']

        # 2. Momentum Score
        momentum_score = self._score_momentum(action, last_bar)
        breakdown['momentum'] = momentum_score
        total_score += momentum_score * self.weights['momentum']

        # 3. Volume Score
        volume_score = self._score_volume(last_bar)
        breakdown['volume'] = volume_score
        total_score += volume_score * self.weights['volume']

        # 4. Volatility Score
        volatility_score = self._score_volatility(last_bar, df)
        breakdown['volatility'] = volatility_score
        total_score += volatility_score * self.weights['volatility']

        # 5. Multi-Timeframe Confirmation
        mtf_score = self._score_mtf_confirmation(action, htf_trend)
        breakdown['mtf_confirmation'] = mtf_score
        total_score += mtf_score * self.weights['mtf_confirmation']

        # 6. Support/Resistance Proximity
        sr_score = self._score_support_resistance(action, last_bar, df)
        breakdown['support_resistance'] = sr_score
        total_score += sr_score * self.weights['support_resistance']

        # Normalize to 10-point scale
        normalized_score = (total_score / self.max_score) * 10

        logger.debug(f"Signal score: {normalized_score:.1f}/10 - {breakdown}")

        return normalized_score, breakdown

    def _score_trend_alignment(self, action: str, last_bar: pd.Series,
                                df: pd.DataFrame) -> float:
        """Score based on trend alignment with multiple MAs"""
        score = 0

        close = last_bar['close']

        # Check FRAMA or EMA alignment
        if 'frama' in last_bar:
            if action == 'LONG' and close > last_bar['frama']:
                score += 0.5
            elif action == 'SHORT' and close < last_bar['frama']:
                score += 0.5

        # Check SuperTrend direction
        if 'supertrend_dir' in last_bar:
            if action == 'LONG' and last_bar['supertrend_dir'] == 1:
                score += 0.5
            elif action == 'SHORT' and last_bar['supertrend_dir'] == -1:
                score += 0.5

        return min(score, 1.0)

    def _score_momentum(self, action: str, last_bar: pd.Series) -> float:
        """Score based on momentum indicators"""
        score = 0

        # RSX/RSI score
        rsx = last_bar.get('rsx', last_bar.get('rsi', 50))
        if action == 'LONG':
            if 30 <= rsx <= 50:
                score += 0.4  # Ideal buy zone
            elif 50 < rsx <= 65:
                score += 0.2  # Still ok
        else:
            if 50 <= rsx <= 70:
                score += 0.4  # Ideal sell zone
            elif 35 <= rsx < 50:
                score += 0.2

        # MACD alignment
        if 'macd' in last_bar and 'macd_signal' in last_bar:
            if action == 'LONG' and last_bar['macd'] > last_bar['macd_signal']:
                score += 0.3
            elif action == 'SHORT' and last_bar['macd'] < last_bar['macd_signal']:
                score += 0.3

        # ADX strength
        adx = last_bar.get('adx', 0)
        if adx >= 25:
            score += 0.3

        return min(score, 1.0)

    def _score_volume(self, last_bar: pd.Series) -> float:
        """Score based on volume confirmation"""
        if 'volume' not in last_bar or 'avg_volume' not in last_bar:
            return 0.5  # Neutral if no volume data

        volume_ratio = last_bar['volume'] / last_bar['avg_volume']

        if volume_ratio >= 2.0:
            return 1.0  # Strong volume
        elif volume_ratio >= 1.5:
            return 0.8
        elif volume_ratio >= 1.2:
            return 0.6
        elif volume_ratio >= 1.0:
            return 0.4
        else:
            return 0.2  # Below average volume

    def _score_volatility(self, last_bar: pd.Series, df: pd.DataFrame) -> float:
        """Score based on volatility conditions"""
        score = 0.5  # Default neutral

        # Check volatility ratio if available
        if 'volatility_ratio' in last_bar:
            vr = last_bar['volatility_ratio']
            if 0.3 <= vr <= 0.7:
                score = 0.8  # Ideal volatility range
            elif vr < 0.3:
                score = 0.4  # Too quiet
            else:
                score = 0.6  # Expanding, could be good for breakouts

        # Check ATR relative to price
        if 'atr' in last_bar:
            atr_pct = (last_bar['atr'] / last_bar['close']) * 100
            if 0.5 <= atr_pct <= 3.0:
                score = max(score, 0.7)  # Healthy volatility

        return score

    def _score_mtf_confirmation(self, action: str, htf_trend: str) -> float:
        """Score based on higher timeframe confirmation"""
        if htf_trend is None:
            return 0.5  # Neutral if no HTF data

        if action == htf_trend:
            return 1.0  # Perfect alignment
        elif htf_trend == 'NEUTRAL':
            return 0.5  # Neutral HTF
        else:
            return 0.0  # Counter-trend

    def _score_support_resistance(self, action: str, last_bar: pd.Series,
                                   df: pd.DataFrame) -> float:
        """Score based on proximity to support/resistance levels"""
        score = 0.5

        close = last_bar['close']

        # Use band indicators for S/R
        if 'iqr_upper' in last_bar and 'iqr_lower' in last_bar:
            upper = last_bar['iqr_upper']
            lower = last_bar['iqr_lower']

            if action == 'LONG':
                # Good if near support
                distance_to_support = (close - lower) / close * 100
                if distance_to_support < 1.0:
                    score = 0.9
                elif distance_to_support < 2.0:
                    score = 0.7

            else:  # SHORT
                # Good if near resistance
                distance_to_resistance = (upper - close) / close * 100
                if distance_to_resistance < 1.0:
                    score = 0.9
                elif distance_to_resistance < 2.0:
                    score = 0.7

        return score

    def passes_threshold(self, score: float) -> bool:
        """Check if score passes minimum threshold"""
        return score >= self.min_score


class MultiTimeframeAnalyzer:
    """
    Analyzes higher timeframes for trend confirmation.
    """

    # Timeframe hierarchy (higher = longer term)
    TIMEFRAME_HIERARCHY = {
        '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
        '1h': 60, '2h': 120, '4h': 240, '6h': 360,
        '12h': 720, '1d': 1440, '1w': 10080
    }

    def __init__(self, data_client, config: dict = None):
        self.data_client = data_client
        self.config = config or {}
        self._htf_cache = {}
        self._cache_ttl = 300  # 5 minutes cache

    def get_higher_timeframe(self, current_tf: str) -> str:
        """Get the next higher timeframe"""
        current_minutes = self.TIMEFRAME_HIERARCHY.get(current_tf, 15)

        htf_mapping = {
            '1m': '5m', '3m': '15m', '5m': '15m',
            '15m': '1h', '30m': '1h', '1h': '4h',
            '2h': '4h', '4h': '1d', '6h': '1d',
            '12h': '1d', '1d': '1w'
        }

        return htf_mapping.get(current_tf, '4h')

    def analyze_htf_trend(self, symbol: str, htf: str = '1h') -> Dict:
        """
        Analyze higher timeframe trend.

        Returns:
            Dict with trend direction and confidence
        """
        cache_key = f"{symbol}_{htf}"
        now = datetime.now()

        # Check cache
        if cache_key in self._htf_cache:
            cached, cached_time = self._htf_cache[cache_key]
            if (now - cached_time).total_seconds() < self._cache_ttl:
                return cached

        try:
            # Fetch HTF data
            df = self.data_client.fetch_ohlcv(symbol, htf, limit=100)

            if df is None or len(df) < 50:
                return {'trend': 'NEUTRAL', 'confidence': 0}

            # Calculate simple trend indicators
            df['ema20'] = ta.ema(df['close'], length=20)
            df['ema50'] = ta.ema(df['close'], length=50)
            df['rsi'] = ta.rsi(df['close'], length=14)

            last = df.iloc[-1]
            prev = df.iloc[-2]

            # Determine trend
            trend = 'NEUTRAL'
            confidence = 0

            # EMA alignment
            if last['close'] > last['ema20'] > last['ema50']:
                trend = 'LONG'
                confidence += 40
            elif last['close'] < last['ema20'] < last['ema50']:
                trend = 'SHORT'
                confidence += 40

            # Price action (higher highs/lows or lower highs/lows)
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

            # RSI confirmation
            rsi = last['rsi']
            if trend == 'LONG' and 40 <= rsi <= 70:
                confidence += 20
            elif trend == 'SHORT' and 30 <= rsi <= 60:
                confidence += 20

            # Cap confidence
            confidence = min(confidence, 100)

            result = {
                'trend': trend,
                'confidence': confidence,
                'ema20': last['ema20'],
                'ema50': last['ema50'],
                'rsi': rsi,
                'timeframe': htf
            }

            # Cache result
            self._htf_cache[cache_key] = (result, now)

            logger.debug(f"HTF analysis {symbol} {htf}: {trend} ({confidence}%)")
            return result

        except Exception as e:
            logger.error(f"Error analyzing HTF for {symbol}: {e}")
            return {'trend': 'NEUTRAL', 'confidence': 0}

    def confirms_signal(self, signal: Dict, htf_analysis: Dict) -> bool:
        """Check if HTF analysis confirms the signal"""
        if htf_analysis['trend'] == 'NEUTRAL':
            return True  # Allow signals when HTF is neutral

        min_confidence = self.config.get('mtf_min_confidence', 50)

        if htf_analysis['confidence'] < min_confidence:
            return True  # Not confident enough to reject

        return signal['action'] == htf_analysis['trend']


class VolumeFilter:
    """Filter signals based on volume conditions"""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.min_volume_ratio = self.config.get('min_volume_ratio', 1.0)
        self.spike_threshold = self.config.get('volume_spike_threshold', 2.0)

    def check_volume(self, df: pd.DataFrame) -> Dict:
        """
        Check volume conditions.

        Returns:
            Dict with volume analysis
        """
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
        """
        Check volatility conditions.

        Returns:
            Dict with volatility analysis
        """
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


class CorrelationFilter:
    """
    Filter to prevent too many correlated signals.
    Limits simultaneous signals on highly correlated pairs.
    """

    # Pre-defined correlation groups (simplified)
    CORRELATION_GROUPS = {
        'btc_related': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
        'altcoins': ['SOL/USDT', 'AVAX/USDT', 'MATIC/USDT'],
        'meme': ['DOGE/USDT', 'SHIB/USDT', 'PEPE/USDT']
    }

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.max_signals_per_group = self.config.get('max_signals_per_group', 2)
        self._active_signals = {}

    def can_signal(self, symbol: str, active_signals: List[str]) -> bool:
        """Check if a new signal is allowed based on correlation limits"""
        symbol_group = self._get_group(symbol)

        if symbol_group is None:
            return True  # Unknown group, allow

        group_count = sum(1 for s in active_signals if self._get_group(s) == symbol_group)

        return group_count < self.max_signals_per_group

    def _get_group(self, symbol: str) -> Optional[str]:
        """Get correlation group for a symbol"""
        for group_name, symbols in self.CORRELATION_GROUPS.items():
            if symbol in symbols:
                return group_name
        return None


class AccuracyFilterManager:
    """
    Manages all accuracy filters and provides unified filtering interface.
    """

    def __init__(self, data_client, config: dict = None):
        self.config = config or {}
        self.data_client = data_client

        # Initialize individual filters
        self.scorer = SignalScorer(config)
        self.mtf_analyzer = MultiTimeframeAnalyzer(data_client, config)
        self.volume_filter = VolumeFilter(config)
        self.volatility_filter = VolatilityFilter(config)
        self.correlation_filter = CorrelationFilter(config)

        # Filter settings
        self.enable_mtf = self.config.get('enable_mtf_filter', True)
        self.enable_volume = self.config.get('enable_volume_filter', True)
        self.enable_volatility = self.config.get('enable_volatility_filter', True)
        self.enable_scoring = self.config.get('enable_signal_scoring', True)
        self.enable_correlation = self.config.get('enable_correlation_filter', False)

        logger.info(f"Accuracy filters initialized: MTF={self.enable_mtf}, "
                   f"Volume={self.enable_volume}, Volatility={self.enable_volatility}, "
                   f"Scoring={self.enable_scoring}")

    def filter_signal(self, signal: Dict, df: pd.DataFrame,
                      timeframe: str = '15m',
                      active_signals: List[str] = None) -> Tuple[bool, Dict]:
        """
        Apply all filters to a signal.

        Returns:
            Tuple of (passes_all_filters, filter_results)
        """
        results = {
            'passes': True,
            'score': None,
            'score_breakdown': None,
            'mtf': None,
            'volume': None,
            'volatility': None,
            'correlation': None,
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
                results['rejection_reason'] = f"HTF trend ({htf_analysis['trend']}) conflicts with signal"
                logger.info(f"Signal rejected: MTF conflict for {symbol}")
                return False, results

        # 2. Volume filter
        if self.enable_volume:
            volume_check = self.volume_filter.check_volume(df)
            results['volume'] = volume_check

            if not volume_check['passes']:
                results['passes'] = False
                results['rejection_reason'] = f"Low volume (ratio: {volume_check['ratio']:.2f})"
                logger.info(f"Signal rejected: Low volume for {symbol}")
                return False, results

        # 3. Volatility filter
        if self.enable_volatility:
            volatility_check = self.volatility_filter.check_volatility(df)
            results['volatility'] = volatility_check

            if not volatility_check['passes']:
                results['passes'] = False
                results['rejection_reason'] = f"Bad volatility ({volatility_check['condition']})"
                logger.info(f"Signal rejected: Volatility issue for {symbol}")
                return False, results

        # 4. Correlation filter
        if self.enable_correlation and active_signals:
            can_signal = self.correlation_filter.can_signal(symbol, active_signals)
            results['correlation'] = {'can_signal': can_signal}

            if not can_signal:
                results['passes'] = False
                results['rejection_reason'] = "Too many correlated signals active"
                logger.info(f"Signal rejected: Correlation limit for {symbol}")
                return False, results

        # 5. Signal scoring
        if self.enable_scoring:
            htf_trend = results['mtf']['trend'] if results['mtf'] else None
            score, breakdown = self.scorer.calculate_score(signal, df, htf_trend)
            results['score'] = score
            results['score_breakdown'] = breakdown

            if not self.scorer.passes_threshold(score):
                results['passes'] = False
                results['rejection_reason'] = f"Low signal score ({score:.1f}/10)"
                logger.info(f"Signal rejected: Low score {score:.1f} for {symbol}")
                return False, results

        logger.info(f"Signal passed all filters: {symbol} (score: {results.get('score', 'N/A')})")
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
