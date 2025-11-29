import pandas as pd
import numpy as np
import pandas_ta as ta
import logging

logger = logging.getLogger(__name__)

class AdvancedIndicatorCalculator:
    """
    Advanced indicator calculator implementing 10 superior indicators:
    1. FRAMA - Fractal Adaptive Moving Average
    2. Jurik RSX - Smoother RSI
    3. SuperTrend - Trend direction
    4. Chandelier Exit - ATR-based stops
    5. Parabolic SAR - Acceleration-based trailing
    6. IQR Bands - Interquartile Range Bands
    7. Kirshenbaum Bands - Linear regression bands
    8. MAD Bands - Mean Absolute Deviation Bands
    9. Volatility Ratio - Squeeze/expansion detector
    10. Klinger Volume Oscillator - Volume confirmation
    """
    
    def __init__(self, config: dict):
        self.config = config
    
    def calculate_frama(self, df: pd.DataFrame, length: int = 16) -> pd.Series:
        """
        Fractal Adaptive Moving Average (FRAMA)
        Adapts to market volatility using fractal dimension
        """
        try:
            close = df['close'].values
            frama = np.zeros(len(close))
            frama[:length] = close[:length]
            
            for i in range(length, len(close)):
                n = length // 2
                
                hl1 = np.max(close[i-length:i-n]) - np.min(close[i-length:i-n])
                hl2 = np.max(close[i-n:i]) - np.min(close[i-n:i])
                hl = np.max(close[i-length:i]) - np.min(close[i-length:i])
                
                if hl1 + hl2 == 0 or hl == 0:
                    alpha = 0.5
                else:
                    d = (np.log(hl1 + hl2) - np.log(hl)) / np.log(2)
                    alpha = np.exp(-4.6 * (d - 1))
                    alpha = np.clip(alpha, 0.01, 1.0)
                
                frama[i] = alpha * close[i] + (1 - alpha) * frama[i-1]
            
            return pd.Series(frama, index=df.index)
        except Exception as e:
            logger.error(f"Error calculating FRAMA: {e}")
            return pd.Series(np.nan, index=df.index)
    
    def calculate_jurik_rsx(self, df: pd.DataFrame, length: int = 14) -> pd.Series:
        """
        Jurik RSX - Smoother, lower-lag version of RSI
        """
        try:
            close = df['close'].values
            
            changes = np.diff(close, prepend=close[0])
            
            gains = np.where(changes > 0, changes, 0)
            losses = np.where(changes < 0, -changes, 0)
            
            alpha = 2 / (length + 1)
            
            avg_gain = np.zeros(len(gains))
            avg_gain[0] = gains[0]
            for i in range(1, len(gains)):
                avg_gain[i] = alpha * gains[i] + (1 - alpha) * avg_gain[i-1]
            
            smooth_gain = np.zeros(len(avg_gain))
            smooth_gain[0] = avg_gain[0]
            for i in range(1, len(avg_gain)):
                smooth_gain[i] = alpha * avg_gain[i] + (1 - alpha) * smooth_gain[i-1]
            
            avg_loss = np.zeros(len(losses))
            avg_loss[0] = losses[0]
            for i in range(1, len(losses)):
                avg_loss[i] = alpha * losses[i] + (1 - alpha) * avg_loss[i-1]
            
            smooth_loss = np.zeros(len(avg_loss))
            smooth_loss[0] = avg_loss[0]
            for i in range(1, len(avg_loss)):
                smooth_loss[i] = alpha * avg_loss[i] + (1 - alpha) * smooth_loss[i-1]
            
            rs = np.where(smooth_loss != 0, smooth_gain / smooth_loss, 100)
            rsx = 100 - (100 / (1 + rs))
            
            return pd.Series(rsx, index=df.index)
        except Exception as e:
            logger.error(f"Error calculating Jurik RSX: {e}")
            return pd.Series(np.nan, index=df.index)
    
    def calculate_supertrend(self, df: pd.DataFrame, length: int = 22, mult: float = 3.0) -> tuple:
        """
        SuperTrend indicator
        Returns: (supertrend, direction) where direction is 1 for long, -1 for short
        """
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            atr = ta.atr(df['high'], df['low'], df['close'], length=length).values
            
            hl2 = (high + low) / 2
            basic_ub = hl2 + mult * atr
            basic_lb = hl2 - mult * atr
            
            final_ub = np.zeros(len(close))
            final_lb = np.zeros(len(close))
            supertrend = np.zeros(len(close))
            direction = np.ones(len(close))
            
            final_ub[0] = basic_ub[0]
            final_lb[0] = basic_lb[0]
            supertrend[0] = final_ub[0]
            direction[0] = -1
            
            for i in range(1, len(close)):
                if basic_ub[i] < final_ub[i-1] or close[i-1] > final_ub[i-1]:
                    final_ub[i] = basic_ub[i]
                else:
                    final_ub[i] = final_ub[i-1]
                
                if basic_lb[i] > final_lb[i-1] or close[i-1] < final_lb[i-1]:
                    final_lb[i] = basic_lb[i]
                else:
                    final_lb[i] = final_lb[i-1]
                
                if supertrend[i-1] == final_ub[i-1]:
                    if close[i] <= final_ub[i]:
                        supertrend[i] = final_ub[i]
                        direction[i] = -1
                    else:
                        supertrend[i] = final_lb[i]
                        direction[i] = 1
                else:
                    if close[i] >= final_lb[i]:
                        supertrend[i] = final_lb[i]
                        direction[i] = 1
                    else:
                        supertrend[i] = final_ub[i]
                        direction[i] = -1
            
            return pd.Series(supertrend, index=df.index), pd.Series(direction, index=df.index)
        except Exception as e:
            logger.error(f"Error calculating SuperTrend: {e}")
            return pd.Series(np.nan, index=df.index), pd.Series(0, index=df.index)
    
    def calculate_chandelier_exit(self, df: pd.DataFrame, length: int = 22, mult: float = 3.0) -> tuple:
        """
        Chandelier Exit - ATR-based stops from price extremes
        Returns: (long_stop, short_stop, direction)
        """
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            atr = ta.atr(df['high'], df['low'], df['close'], length=length).values
            
            long_stop = np.zeros(len(close))
            short_stop = np.zeros(len(close))
            direction = np.ones(len(close))
            
            for i in range(length, len(close)):
                highest = np.max(high[i-length+1:i+1])
                long_stop[i] = highest - mult * atr[i]
                
                if i > length and close[i-1] > long_stop[i-1]:
                    long_stop[i] = max(long_stop[i], long_stop[i-1])
                
                lowest = np.min(low[i-length+1:i+1])
                short_stop[i] = lowest + mult * atr[i]
                
                if i > length and close[i-1] < short_stop[i-1]:
                    short_stop[i] = min(short_stop[i], short_stop[i-1])
                
                if close[i] > short_stop[i-1]:
                    direction[i] = 1
                elif close[i] < long_stop[i-1]:
                    direction[i] = -1
                else:
                    direction[i] = direction[i-1]
            
            return (pd.Series(long_stop, index=df.index), 
                    pd.Series(short_stop, index=df.index),
                    pd.Series(direction, index=df.index))
        except Exception as e:
            logger.error(f"Error calculating Chandelier Exit: {e}")
            return (pd.Series(np.nan, index=df.index), 
                    pd.Series(np.nan, index=df.index),
                    pd.Series(0, index=df.index))
    
    def calculate_iqr_bands(self, df: pd.DataFrame, length: int = 20, mult: float = 1.5) -> tuple:
        """
        Interquartile Range Bands - Statistical bands using quartiles
        Returns: (upper, basis, lower)
        """
        try:
            close = df['close'].values
            
            upper = np.zeros(len(close))
            basis = np.zeros(len(close))
            lower = np.zeros(len(close))
            
            for i in range(length, len(close)):
                window = close[i-length+1:i+1]
                
                q1 = np.percentile(window, 25)
                q2 = np.percentile(window, 50)  # median
                q3 = np.percentile(window, 75)
                
                iqr = q3 - q1
                
                basis[i] = q2
                upper[i] = q3 + mult * iqr
                lower[i] = q1 - mult * iqr
            
            return (pd.Series(upper, index=df.index),
                    pd.Series(basis, index=df.index),
                    pd.Series(lower, index=df.index))
        except Exception as e:
            logger.error(f"Error calculating IQR Bands: {e}")
            return (pd.Series(np.nan, index=df.index),
                    pd.Series(np.nan, index=df.index),
                    pd.Series(np.nan, index=df.index))
    
    def calculate_kirshenbaum_bands(self, df: pd.DataFrame, length: int = 20, mult: float = 2.0) -> tuple:
        """
        Kirshenbaum Bands - Linear regression with standard error
        Returns: (upper, basis, lower)
        """
        try:
            close = df['close'].values
            
            upper = np.zeros(len(close))
            basis = np.zeros(len(close))
            lower = np.zeros(len(close))
            
            for i in range(length, len(close)):
                window = close[i-length+1:i+1]
                x = np.arange(length)
                
                coeffs = np.polyfit(x, window, 1)
                linreg = np.polyval(coeffs, length - 1)
                
                fitted = np.polyval(coeffs, x)
                residuals = window - fitted
                std_err = np.std(residuals)
                
                basis[i] = linreg
                upper[i] = linreg + mult * std_err
                lower[i] = linreg - mult * std_err
            
            return (pd.Series(upper, index=df.index),
                    pd.Series(basis, index=df.index),
                    pd.Series(lower, index=df.index))
        except Exception as e:
            logger.error(f"Error calculating Kirshenbaum Bands: {e}")
            return (pd.Series(np.nan, index=df.index),
                    pd.Series(np.nan, index=df.index),
                    pd.Series(np.nan, index=df.index))
    
    def calculate_mad_bands(self, df: pd.DataFrame, length: int = 20, mult: float = 2.0) -> tuple:
        """
        Mean Absolute Deviation Bands - More robust than standard deviation
        Returns: (upper, basis, lower)
        """
        try:
            close = df['close'].values
            
            upper = np.zeros(len(close))
            basis = np.zeros(len(close))
            lower = np.zeros(len(close))
            
            for i in range(length, len(close)):
                window = close[i-length+1:i+1]
                
                mean = np.mean(window)
                
                mad = np.mean(np.abs(window - mean))
                
                basis[i] = mean
                upper[i] = mean + mult * mad
                lower[i] = mean - mult * mad
            
            return (pd.Series(upper, index=df.index),
                    pd.Series(basis, index=df.index),
                    pd.Series(lower, index=df.index))
        except Exception as e:
            logger.error(f"Error calculating MAD Bands: {e}")
            return (pd.Series(np.nan, index=df.index),
                    pd.Series(np.nan, index=df.index),
                    pd.Series(np.nan, index=df.index))
    
    def calculate_volatility_ratio(self, df: pd.DataFrame, short_length: int = 5, long_length: int = 20) -> pd.Series:
        """
        Volatility Ratio - Detects volatility expansion/contraction
        Low values indicate squeeze, high values indicate expansion
        """
        try:
            high = df['high'].values
            low = df['low'].values
            
            vr = np.zeros(len(high))
            
            for i in range(long_length, len(high)):
                short_range = np.max(high[i-short_length+1:i+1]) - np.min(low[i-short_length+1:i+1])
                
                long_range = np.max(high[i-long_length+1:i+1]) - np.min(low[i-long_length+1:i+1])
                
                if long_range != 0:
                    vr[i] = short_range / long_range
                else:
                    vr[i] = 1.0
            
            return pd.Series(vr, index=df.index)
        except Exception as e:
            logger.error(f"Error calculating Volatility Ratio: {e}")
            return pd.Series(np.nan, index=df.index)
    
    def calculate_klinger(self, df: pd.DataFrame, fast: int = 34, slow: int = 55, signal: int = 13) -> tuple:
        """
        Klinger Volume Oscillator - Volume-weighted momentum
        Returns: (klinger, signal_line)
        """
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            volume = df['volume'].values
            
            tp = (high + low + close) / 3
            
            trend = np.zeros(len(close))
            for i in range(1, len(close)):
                if tp[i] > tp[i-1]:
                    trend[i] = 1
                else:
                    trend[i] = -1
            
            # Fix: Prevent division by zero when high + low = 0
            hl_sum = high + low
            hl_sum = np.where(hl_sum == 0, 1e-10, hl_sum)  # Replace zeros with small value
            vf = volume * trend * np.abs(2 * ((high - low) / hl_sum) - 1) * 100
            
            fast_ema = pd.Series(vf).ewm(span=fast, adjust=False).mean().values
            slow_ema = pd.Series(vf).ewm(span=slow, adjust=False).mean().values
            
            klinger = fast_ema - slow_ema
            signal_line = pd.Series(klinger).ewm(span=signal, adjust=False).mean().values
            
            return pd.Series(klinger, index=df.index), pd.Series(signal_line, index=df.index)
        except Exception as e:
            logger.error(f"Error calculating Klinger: {e}")
            return pd.Series(np.nan, index=df.index), pd.Series(np.nan, index=df.index)
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all advanced indicators
        """
        if df is None or len(df) < 200:
            logger.warning("Insufficient data for indicator calculation")
            return None
        
        df = df.copy()
        
        try:
            strategy = self.config['strategy']
            
            frama_length = strategy.get('frama_length', 16)
            df['frama'] = self.calculate_frama(df, frama_length)
            
            rsx_length = strategy.get('rsx_length', 14)
            df['rsx'] = self.calculate_jurik_rsx(df, rsx_length)
            
            st_length = strategy.get('supertrend_length', 22)
            st_mult = strategy.get('supertrend_mult', 3.0)
            df['supertrend'], df['supertrend_dir'] = self.calculate_supertrend(df, st_length, st_mult)
            
            ce_length = strategy.get('chandelier_length', 22)
            ce_mult = strategy.get('chandelier_mult', 3.0)
            df['ce_long'], df['ce_short'], df['ce_dir'] = self.calculate_chandelier_exit(df, ce_length, ce_mult)
            
            psar_start = strategy.get('psar_start', 0.02)
            psar_max = strategy.get('psar_max', 0.2)
            psar = ta.psar(df['high'], df['low'], df['close'], af0=psar_start, af=psar_start, max_af=psar_max)
            df['psar'] = psar[f'PSARl_{psar_start}_{psar_max}'].fillna(psar[f'PSARs_{psar_start}_{psar_max}'])
            df['psar_dir'] = np.where(df['close'] > df['psar'], 1, -1)
            
            iqr_length = strategy.get('iqr_length', 20)
            iqr_mult = strategy.get('iqr_mult', 1.5)
            df['iqr_upper'], df['iqr_basis'], df['iqr_lower'] = self.calculate_iqr_bands(df, iqr_length, iqr_mult)
            
            kb_length = strategy.get('kb_length', 20)
            kb_mult = strategy.get('kb_mult', 2.0)
            df['kb_upper'], df['kb_basis'], df['kb_lower'] = self.calculate_kirshenbaum_bands(df, kb_length, kb_mult)
            
            mad_length = strategy.get('mad_length', 20)
            mad_mult = strategy.get('mad_mult', 2.0)
            df['mad_upper'], df['mad_basis'], df['mad_lower'] = self.calculate_mad_bands(df, mad_length, mad_mult)
            
            vr_short = strategy.get('vr_short', 5)
            vr_long = strategy.get('vr_long', 20)
            df['volatility_ratio'] = self.calculate_volatility_ratio(df, vr_short, vr_long)
            
            klinger_fast = strategy.get('klinger_fast', 34)
            klinger_slow = strategy.get('klinger_slow', 55)
            klinger_signal = strategy.get('klinger_signal', 13)
            df['klinger'], df['klinger_signal'] = self.calculate_klinger(df, klinger_fast, klinger_slow, klinger_signal)
            
            # Fix: Use dynamic column name extraction instead of hardcoded names
            adx_length = strategy.get('adx_length', 14)
            adx_result = ta.adx(df['high'], df['low'], df['close'], length=adx_length)
            adx_col = [col for col in adx_result.columns if col.startswith('ADX_')][0]
            df['adx'] = adx_result[adx_col]

            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=strategy.get('atr_length', 14))

            macd_fast = strategy.get('macd_fast', 12)
            macd_slow = strategy.get('macd_slow', 26)
            macd_signal_len = strategy.get('macd_signal', 9)
            macd_result = ta.macd(df['close'], fast=macd_fast, slow=macd_slow, signal=macd_signal_len)
            macd_col = [col for col in macd_result.columns if col.startswith('MACD_') and not col.startswith('MACDs') and not col.startswith('MACDh')][0]
            macd_signal_col = [col for col in macd_result.columns if col.startswith('MACDs_')][0]
            df['macd'] = macd_result[macd_col]
            df['macd_signal'] = macd_result[macd_signal_col]
            
            df['avg_volume'] = ta.sma(df['volume'], length=strategy.get('volume_period', 20))
            
            df.dropna(inplace=True)

            # Fix: Check for data insufficiency after dropna
            if len(df) < 50:
                logger.warning(f"Insufficient data after indicator calculation: {len(df)} bars remaining")
                return None

            logger.debug(f"Calculated all advanced indicators for {len(df)} bars")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating advanced indicators: {e}")
            return None
