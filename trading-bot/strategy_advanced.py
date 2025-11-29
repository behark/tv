import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

class AdvancedConfluenceStrategy:
    """
    Advanced Confluence Strategy using 10 superior indicators:
    
    Trend Following Mode:
    - FRAMA for adaptive trend baseline
    - SuperTrend for direction changes
    - Chandelier Exit for stops
    - Parabolic SAR for trailing (when ADX strong)
    
    Mean Reversion Mode:
    - IQR/MAD Bands for extreme levels
    - Jurik RSX for momentum
    - Low ADX filter
    
    Breakout Mode:
    - Kirshenbaum Bands for contraction/expansion
    - Volatility Ratio for squeeze detection
    - Volume confirmation
    
    Universal Filters:
    - ADX for regime detection
    - Klinger for volume confirmation
    - MACD for additional confirmation
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.last_bar_time = {}
        self.last_signal = {}
        self.strategy_mode = config.get('strategy_mode', 'trend_following')  # trend_following, mean_reversion, breakout, adaptive
    
    def check_trend_following_signals(self, symbol: str, last_bar: pd.Series, prev_bar: pd.Series) -> Optional[Dict]:
        """
        Trend Following Strategy using FRAMA + SuperTrend + Chandelier Exit
        """
        try:
            strategy = self.config['strategy']
            weights = strategy['weights']
            
            close = last_bar['close']
            frama = last_bar['frama']
            rsx = last_bar['rsx']
            rsx_prev = prev_bar['rsx']
            supertrend_dir = last_bar['supertrend_dir']
            supertrend_dir_prev = prev_bar['supertrend_dir']
            adx = last_bar['adx']
            klinger = last_bar['klinger']
            klinger_signal = last_bar['klinger_signal']
            macd = last_bar['macd']
            macd_signal = last_bar['macd_signal']
            atr = last_bar['atr']
            ce_long = last_bar['ce_long']
            ce_short = last_bar['ce_short']
            
            trend_up = close > frama
            trend_down = close < frama
            
            st_buy_signal = supertrend_dir == 1 and supertrend_dir_prev == -1
            st_sell_signal = supertrend_dir == -1 and supertrend_dir_prev == 1
            
            rsx_bullish = rsx < 50 and rsx > 30
            rsx_bearish = rsx > 50 and rsx < 70
            rsx_cross_up = rsx_prev <= 30 and rsx > 30
            rsx_cross_down = rsx_prev >= 70 and rsx < 70
            
            macd_bullish = macd > macd_signal
            macd_bearish = macd < macd_signal
            
            adx_threshold = strategy.get('adx_threshold', 20)
            adx_strong = adx > adx_threshold
            
            klinger_bullish = klinger > klinger_signal
            klinger_bearish = klinger < klinger_signal
            
            long_score = 0
            short_score = 0
            
            if trend_up:
                long_score += weights.get('trend', 1.0)
            if trend_down:
                short_score += weights.get('trend', 1.0)
            
            if st_buy_signal:
                long_score += weights.get('supertrend', 2.0)  # Higher weight for primary trigger
            if st_sell_signal:
                short_score += weights.get('supertrend', 2.0)
            
            if rsx_cross_up or rsx_bullish:
                long_score += weights.get('rsx', 1.0)
            if rsx_cross_down or rsx_bearish:
                short_score += weights.get('rsx', 1.0)
            
            if macd_bullish:
                long_score += weights.get('macd', 1.0)
            if macd_bearish:
                short_score += weights.get('macd', 1.0)
            
            if adx_strong:
                long_score += weights.get('adx', 1.0)
                short_score += weights.get('adx', 1.0)
            
            if klinger_bullish:
                long_score += weights.get('volume', 0.5)
            if klinger_bearish:
                short_score += weights.get('volume', 0.5)
            
            score_threshold = strategy.get('score_threshold', 3.0)
            
            if long_score >= score_threshold:
                return {
                    'action': 'LONG',
                    'entry_price': close,
                    'stop_loss': ce_long,
                    'take_profit': close + (close - ce_long) * strategy.get('risk_reward_ratio', 1.5),
                    'score': long_score,
                    'indicators': {
                        'frama': frama,
                        'rsx': rsx,
                        'supertrend_dir': supertrend_dir,
                        'adx': adx,
                        'klinger': klinger
                    }
                }
            
            if short_score >= score_threshold:
                return {
                    'action': 'SHORT',
                    'entry_price': close,
                    'stop_loss': ce_short,
                    'take_profit': close - (ce_short - close) * strategy.get('risk_reward_ratio', 1.5),
                    'score': short_score,
                    'indicators': {
                        'frama': frama,
                        'rsx': rsx,
                        'supertrend_dir': supertrend_dir,
                        'adx': adx,
                        'klinger': klinger
                    }
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in trend following signals: {e}")
            return None
    
    def check_mean_reversion_signals(self, symbol: str, last_bar: pd.Series, prev_bar: pd.Series) -> Optional[Dict]:
        """
        Mean Reversion Strategy using IQR/MAD Bands + Jurik RSX
        """
        try:
            strategy = self.config['strategy']
            
            close = last_bar['close']
            rsx = last_bar['rsx']
            iqr_upper = last_bar['iqr_upper']
            iqr_lower = last_bar['iqr_lower']
            iqr_basis = last_bar['iqr_basis']
            mad_upper = last_bar['mad_upper']
            mad_lower = last_bar['mad_lower']
            adx = last_bar['adx']
            atr = last_bar['atr']
            
            use_iqr = strategy.get('use_iqr_bands', True)
            upper = iqr_upper if use_iqr else mad_upper
            lower = iqr_lower if use_iqr else mad_lower
            basis = iqr_basis if use_iqr else last_bar['mad_basis']
            
            # Fix: Use >= instead of > for ADX threshold comparison
            adx_threshold = strategy.get('mr_adx_threshold', 20)
            if adx >= adx_threshold:
                return None  # Skip mean reversion in trending markets
            
            rsx_oversold = strategy.get('rsx_oversold', 30)
            rsx_overbought = strategy.get('rsx_overbought', 70)
            
            if close <= lower and rsx < rsx_oversold:
                atr_mult = strategy.get('mr_atr_mult', 1.0)
                return {
                    'action': 'LONG',
                    'entry_price': close,
                    'stop_loss': close - atr * atr_mult,
                    'take_profit': basis,  # Exit at midline
                    'score': 3.0,  # Fixed score for MR
                    'indicators': {
                        'rsx': rsx,
                        'band_lower': lower,
                        'adx': adx
                    }
                }
            
            if close >= upper and rsx > rsx_overbought:
                atr_mult = strategy.get('mr_atr_mult', 1.0)
                return {
                    'action': 'SHORT',
                    'entry_price': close,
                    'stop_loss': close + atr * atr_mult,
                    'take_profit': basis,  # Exit at midline
                    'score': 3.0,  # Fixed score for MR
                    'indicators': {
                        'rsx': rsx,
                        'band_upper': upper,
                        'adx': adx
                    }
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in mean reversion signals: {e}")
            return None
    
    def check_breakout_signals(self, symbol: str, last_bar: pd.Series, prev_bar: pd.Series) -> Optional[Dict]:
        """
        Breakout Strategy using Kirshenbaum Bands + Volatility Ratio
        """
        try:
            strategy = self.config['strategy']
            
            close = last_bar['close']
            close_prev = prev_bar['close']
            kb_upper = last_bar['kb_upper']
            kb_lower = last_bar['kb_lower']
            kb_upper_prev = prev_bar['kb_upper']
            kb_lower_prev = prev_bar['kb_lower']
            volatility_ratio = last_bar['volatility_ratio']
            adx = last_bar['adx']
            adx_prev = prev_bar['adx']
            volume = last_bar['volume']
            avg_volume = last_bar['avg_volume']
            macd = last_bar['macd']
            macd_signal = last_bar['macd_signal']
            macd_prev = prev_bar['macd']
            macd_signal_prev = prev_bar['macd_signal']
            atr = last_bar['atr']
            ce_long = last_bar['ce_long']
            ce_short = last_bar['ce_short']
            
            vr_threshold = strategy.get('vr_threshold', 0.5)
            vr_expanding = volatility_ratio > vr_threshold
            
            adx_rising = adx > adx_prev
            
            volume_mult = strategy.get('breakout_volume_mult', 1.5)
            volume_spike = volume > avg_volume * volume_mult
            
            macd_cross_up = macd_prev <= macd_signal_prev and macd > macd_signal
            macd_cross_down = macd_prev >= macd_signal_prev and macd < macd_signal
            
            if (close > kb_upper and close_prev <= kb_upper_prev and
                vr_expanding and volume_spike and (macd_cross_up or adx_rising)):
                
                return {
                    'action': 'LONG',
                    'entry_price': close,
                    'stop_loss': ce_long,
                    'take_profit': close + (close - ce_long) * strategy.get('risk_reward_ratio', 2.0),
                    'score': 4.0,  # High confidence breakout
                    'indicators': {
                        'kb_upper': kb_upper,
                        'volatility_ratio': volatility_ratio,
                        'adx': adx,
                        'volume': volume
                    }
                }
            
            if (close < kb_lower and close_prev >= kb_lower_prev and
                vr_expanding and volume_spike and (macd_cross_down or adx_rising)):
                
                return {
                    'action': 'SHORT',
                    'entry_price': close,
                    'stop_loss': ce_short,
                    'take_profit': close - (ce_short - close) * strategy.get('risk_reward_ratio', 2.0),
                    'score': 4.0,  # High confidence breakout
                    'indicators': {
                        'kb_lower': kb_lower,
                        'volatility_ratio': volatility_ratio,
                        'adx': adx,
                        'volume': volume
                    }
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in breakout signals: {e}")
            return None
    
    def check_signals(self, symbol: str, df: pd.DataFrame) -> Optional[Dict]:
        """
        Main signal checking function - routes to appropriate strategy mode
        """
        if df is None or len(df) < 2:
            return None
        
        try:
            last_bar = df.iloc[-1]
            prev_bar = df.iloc[-2]
            bar_time = df.index[-1]
            
            if symbol in self.last_bar_time and self.last_bar_time[symbol] == bar_time:
                return None
            
            self.last_bar_time[symbol] = bar_time
            
            signal = None
            
            if self.strategy_mode == 'trend_following':
                signal = self.check_trend_following_signals(symbol, last_bar, prev_bar)
            elif self.strategy_mode == 'mean_reversion':
                signal = self.check_mean_reversion_signals(symbol, last_bar, prev_bar)
            elif self.strategy_mode == 'breakout':
                signal = self.check_breakout_signals(symbol, last_bar, prev_bar)
            elif self.strategy_mode == 'adaptive':
                tf_signal = self.check_trend_following_signals(symbol, last_bar, prev_bar)
                mr_signal = self.check_mean_reversion_signals(symbol, last_bar, prev_bar)
                bo_signal = self.check_breakout_signals(symbol, last_bar, prev_bar)
                
                signals = [s for s in [tf_signal, mr_signal, bo_signal] if s is not None]
                if signals:
                    signal = max(signals, key=lambda x: x['score'])
            
            if signal:
                signal_key = f"{symbol}_{signal['action']}"
                if signal_key in self.last_signal and self.last_signal[signal_key] == bar_time:
                    return None
                
                self.last_signal[signal_key] = bar_time
                
                signal['symbol'] = symbol
                signal['timestamp'] = bar_time
                signal['max_score'] = sum(self.config['strategy']['weights'].values())
                signal['strategy_mode'] = self.strategy_mode
                
                signal['rsx'] = last_bar['rsx']
                signal['adx'] = last_bar['adx']
                signal['atr'] = last_bar['atr']
                
                logger.info(f"{signal['action']} signal for {symbol} at {signal['entry_price']:.6f} "
                           f"(score: {signal['score']:.1f}, mode: {self.strategy_mode})")
                
                return signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking signals for {symbol}: {e}")
            return None
