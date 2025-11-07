import pandas as pd
import logging
from typing import Optional, Dict, Tuple

logger = logging.getLogger(__name__)

class ConfluenceStrategy:
    def __init__(self, config: dict):
        self.config = config
        self.last_bar_time = {}
        self.last_signal = {}
    
    def check_signals(self, symbol: str, df: pd.DataFrame) -> Optional[Dict]:
        if df is None or len(df) < 2:
            return None
        
        try:
            last_bar = df.iloc[-1]
            prev_bar = df.iloc[-2]
            
            bar_time = df.index[-1]
            
            if symbol in self.last_bar_time and self.last_bar_time[symbol] == bar_time:
                return None
            
            self.last_bar_time[symbol] = bar_time
            
            close = last_bar['close']
            ema = last_bar['ema']
            rsi = last_bar['rsi']
            rsi_prev = prev_bar['rsi']
            macd = last_bar['macd']
            macd_signal = last_bar['macd_signal']
            macd_prev = prev_bar['macd']
            macd_signal_prev = prev_bar['macd_signal']
            adx = last_bar['adx']
            volume = last_bar['volume']
            avg_volume = last_bar['avg_volume']
            atr = last_bar['atr']
            
            trend_up = close > ema
            trend_down = close < ema
            
            rsi_oversold = self.config['strategy']['rsi_oversold']
            rsi_overbought = self.config['strategy']['rsi_overbought']
            
            rsi_cross_up = rsi_prev <= rsi_oversold and rsi > rsi_oversold
            rsi_cross_down = rsi_prev >= rsi_overbought and rsi < rsi_overbought
            rsi_bullish = rsi < 50 and rsi > rsi_oversold
            rsi_bearish = rsi > 50 and rsi < rsi_overbought
            
            macd_cross_up = macd_prev <= macd_signal_prev and macd > macd_signal
            macd_cross_down = macd_prev >= macd_signal_prev and macd < macd_signal
            macd_bullish = macd > macd_signal
            macd_bearish = macd < macd_signal
            
            adx_threshold = self.config['strategy']['adx_threshold']
            adx_strong = adx > adx_threshold
            
            volume_multiplier = self.config['strategy']['volume_multiplier']
            volume_high = volume > avg_volume * volume_multiplier
            
            weights = self.config['strategy']['weights']
            use_trend_filter = self.config['strategy']['use_trend_filter']
            use_adx_filter = self.config['strategy']['use_adx_filter']
            use_volume_filter = self.config['strategy']['use_volume_filter']
            
            long_score_trend = (weights['trend'] if trend_up else 0) if use_trend_filter else weights['trend']
            long_score_rsi = weights['rsi'] if (rsi_cross_up or rsi_bullish) else 0
            long_score_macd = weights['macd'] if (macd_cross_up or macd_bullish) else 0
            long_score_adx = (weights['adx'] if adx_strong else 0) if use_adx_filter else weights['adx']
            long_score_volume = (weights['volume'] if volume_high else 0) if use_volume_filter else weights['volume']
            
            long_score = long_score_trend + long_score_rsi + long_score_macd + long_score_adx + long_score_volume
            
            short_score_trend = (weights['trend'] if trend_down else 0) if use_trend_filter else weights['trend']
            short_score_rsi = weights['rsi'] if (rsi_cross_down or rsi_bearish) else 0
            short_score_macd = weights['macd'] if (macd_cross_down or macd_bearish) else 0
            short_score_adx = (weights['adx'] if adx_strong else 0) if use_adx_filter else weights['adx']
            short_score_volume = (weights['volume'] if volume_high else 0) if use_volume_filter else weights['volume']
            
            short_score = short_score_trend + short_score_rsi + short_score_macd + short_score_adx + short_score_volume
            
            score_threshold = self.config['strategy']['score_threshold']
            enable_long = self.config['enable_long']
            enable_short = self.config['enable_short']
            
            atr_multiplier_sl = self.config['strategy']['atr_multiplier_sl']
            risk_reward_ratio = self.config['strategy']['risk_reward_ratio']
            
            signal = None
            
            if enable_long and long_score >= score_threshold:
                last_signal_key = f"{symbol}_LONG"
                if last_signal_key not in self.last_signal or self.last_signal[last_signal_key] != bar_time:
                    self.last_signal[last_signal_key] = bar_time
                    
                    entry_price = close
                    stop_loss = entry_price - (atr * atr_multiplier_sl)
                    take_profit = entry_price + (atr * atr_multiplier_sl * risk_reward_ratio)
                    
                    signal = {
                        'symbol': symbol,
                        'action': 'LONG',
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'score': long_score,
                        'max_score': sum(weights.values()),
                        'rsi': rsi,
                        'adx': adx,
                        'macd': macd,
                        'atr': atr,
                        'timestamp': bar_time
                    }
                    
                    logger.info(f"LONG signal for {symbol} at {entry_price:.4f} (score: {long_score:.1f})")
            
            elif enable_short and short_score >= score_threshold:
                last_signal_key = f"{symbol}_SHORT"
                if last_signal_key not in self.last_signal or self.last_signal[last_signal_key] != bar_time:
                    self.last_signal[last_signal_key] = bar_time
                    
                    entry_price = close
                    stop_loss = entry_price + (atr * atr_multiplier_sl)
                    take_profit = entry_price - (atr * atr_multiplier_sl * risk_reward_ratio)
                    
                    signal = {
                        'symbol': symbol,
                        'action': 'SHORT',
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'score': short_score,
                        'max_score': sum(weights.values()),
                        'rsi': rsi,
                        'adx': adx,
                        'macd': macd,
                        'atr': atr,
                        'timestamp': bar_time
                    }
                    
                    logger.info(f"SHORT signal for {symbol} at {entry_price:.4f} (score: {short_score:.1f})")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error checking signals for {symbol}: {e}")
            return None
