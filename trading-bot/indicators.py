import pandas as pd
import pandas_ta as ta
import logging

logger = logging.getLogger(__name__)

class IndicatorCalculator:
    def __init__(self, config: dict):
        self.config = config
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or len(df) < 200:
            logger.warning("Insufficient data for indicator calculation")
            return None
        
        df = df.copy()
        
        try:
            ema_length = self.config['strategy']['ema_length']
            df['ema'] = ta.ema(df['close'], length=ema_length)
            
            rsi_length = self.config['strategy']['rsi_length']
            df['rsi'] = ta.rsi(df['close'], length=rsi_length)
            
            macd_fast = self.config['strategy']['macd_fast']
            macd_slow = self.config['strategy']['macd_slow']
            macd_signal = self.config['strategy']['macd_signal']
            macd = ta.macd(df['close'], fast=macd_fast, slow=macd_slow, signal=macd_signal)
            df['macd'] = macd[f'MACD_{macd_fast}_{macd_slow}_{macd_signal}']
            df['macd_signal'] = macd[f'MACDs_{macd_fast}_{macd_slow}_{macd_signal}']
            df['macd_hist'] = macd[f'MACDh_{macd_fast}_{macd_slow}_{macd_signal}']
            
            adx_length = self.config['strategy']['adx_length']
            adx = ta.adx(df['high'], df['low'], df['close'], length=adx_length)
            df['adx'] = adx[f'ADX_{adx_length}']
            df['di_plus'] = adx[f'DMP_{adx_length}']
            df['di_minus'] = adx[f'DMN_{adx_length}']
            
            atr_length = self.config['strategy']['atr_length']
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=atr_length)
            
            volume_period = self.config['strategy']['volume_period']
            df['avg_volume'] = ta.sma(df['volume'], length=volume_period)
            
            df.dropna(inplace=True)

            # Fix: Check for data insufficiency after dropna
            if len(df) < 50:
                logger.warning(f"Insufficient data after indicator calculation: {len(df)} bars remaining")
                return None

            logger.debug(f"Calculated indicators for {len(df)} bars")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return None
