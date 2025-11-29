import ccxt
import pandas as pd
import logging
from typing import Optional, List, Dict
import time
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Market cache refresh interval (in seconds)
MARKET_CACHE_TTL = 3600  # 1 hour

class DataClient:
    def __init__(self, exchange_name: str = 'mexc', exchange_type: str = 'futures', testnet: bool = False, timeout: int = 30000):
        self.exchange_name = exchange_name
        self.exchange_type = exchange_type
        self._markets_loaded_at = None

        try:
            exchange_class = getattr(ccxt, exchange_name)
            self.exchange = exchange_class({
                'enableRateLimit': True,
                'timeout': timeout,  # Fix: Add explicit timeout (in milliseconds)
                'options': {
                    'defaultType': exchange_type,
                }
            })

            if testnet:
                self.exchange.set_sandbox_mode(True)

            logger.info(f"Initialized {exchange_name} {exchange_type} exchange with {timeout}ms timeout")

        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            raise
    
    def load_markets(self, force: bool = False) -> bool:
        """Load markets with caching support"""
        try:
            # Fix: Implement market cache refresh
            if not force and self._markets_loaded_at:
                elapsed = (datetime.now() - self._markets_loaded_at).total_seconds()
                if elapsed < MARKET_CACHE_TTL:
                    logger.debug(f"Using cached markets (age: {elapsed:.0f}s)")
                    return True

            self.exchange.load_markets()
            self._markets_loaded_at = datetime.now()
            logger.info(f"Loaded {len(self.exchange.markets)} markets from {self.exchange_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to load markets: {e}")
            return False

    def refresh_markets(self) -> bool:
        """Force refresh markets cache"""
        return self.load_markets(force=True)
    
    def check_symbol_exists(self, symbol: str) -> bool:
        """Check if symbol exists, with automatic cache refresh"""
        try:
            if not hasattr(self.exchange, 'markets') or not self.exchange.markets:
                self.load_markets()
            else:
                # Fix: Refresh cache if TTL expired
                self.load_markets()

            return symbol in self.exchange.markets
        except Exception as e:
            logger.error(f"Error checking symbol {symbol}: {e}")
            return False
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = '15m', limit: int = 300) -> Optional[pd.DataFrame]:
        try:
            if not self.check_symbol_exists(symbol):
                logger.warning(f"Symbol {symbol} not found on {self.exchange_name}")
                return None
            
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv:
                logger.warning(f"No data returned for {symbol}")
                return None
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            logger.debug(f"Fetched {len(df)} bars for {symbol} on {timeframe}")
            
            return df
            
        except ccxt.NetworkError as e:
            logger.error(f"Network error fetching {symbol}: {e}")
            return None
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error fetching {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching {symbol}: {e}")
            return None
    
    def get_available_pairs(self, pairs: List[str]) -> Dict[str, bool]:
        if not hasattr(self.exchange, 'markets') or not self.exchange.markets:
            self.load_markets()
        
        available = {}
        for pair in pairs:
            available[pair] = self.check_symbol_exists(pair)
            if available[pair]:
                logger.info(f"✓ {pair} is available on {self.exchange_name}")
            else:
                logger.warning(f"✗ {pair} is NOT available on {self.exchange_name}")
        
        return available
