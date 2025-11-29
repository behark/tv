#!/usr/bin/env python3
"""
Enhanced Data Client for Multi-Tier Trading Bots
Supports OHLCV, funding rates, open interest, and order book data.
"""

import ccxt
import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Market cache refresh interval (in seconds)
MARKET_CACHE_TTL = 3600  # 1 hour


class DataClient:
    """
    Enhanced data client with support for:
    - OHLCV data fetching
    - Multiple timeframes
    - Funding rates (crypto futures)
    - Open interest
    - Order book data
    - Market cache management
    """

    def __init__(self, exchange_name: str = 'mexc', exchange_type: str = 'futures',
                 testnet: bool = False, timeout: int = 30000):
        self.exchange_name = exchange_name
        self.exchange_type = exchange_type
        self._markets_loaded_at = None
        self._funding_cache = {}
        self._oi_cache = {}

        try:
            exchange_class = getattr(ccxt, exchange_name)
            self.exchange = exchange_class({
                'enableRateLimit': True,
                'timeout': timeout,
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
                self.load_markets()

            return symbol in self.exchange.markets
        except Exception as e:
            logger.error(f"Error checking symbol {symbol}: {e}")
            return False

    def fetch_ohlcv(self, symbol: str, timeframe: str = '15m',
                    limit: int = 300) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data for a symbol"""
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

    def fetch_multiple_timeframes(self, symbol: str,
                                   timeframes: List[str] = ['15m', '1h', '4h'],
                                   limit: int = 200) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV data for multiple timeframes"""
        result = {}
        for tf in timeframes:
            df = self.fetch_ohlcv(symbol, tf, limit)
            if df is not None:
                result[tf] = df
        return result

    def fetch_funding_rate(self, symbol: str) -> Optional[Dict]:
        """
        Fetch current funding rate for a futures symbol.
        Returns dict with 'rate', 'timestamp', 'next_funding_time'
        """
        try:
            # Check cache (funding rate doesn't change frequently)
            cache_key = symbol
            if cache_key in self._funding_cache:
                cached, cached_time = self._funding_cache[cache_key]
                if (datetime.now() - cached_time).total_seconds() < 60:  # 1 min cache
                    return cached

            if hasattr(self.exchange, 'fetch_funding_rate'):
                funding = self.exchange.fetch_funding_rate(symbol)
                result = {
                    'symbol': symbol,
                    'rate': funding.get('fundingRate', 0),
                    'timestamp': funding.get('timestamp'),
                    'next_funding_time': funding.get('fundingTimestamp'),
                    'rate_percent': funding.get('fundingRate', 0) * 100
                }
                self._funding_cache[cache_key] = (result, datetime.now())
                return result
            else:
                # Try alternative method
                if hasattr(self.exchange, 'fapiPublicGetPremiumIndex'):
                    # Binance-style
                    pass
                logger.debug(f"Funding rate not available for {self.exchange_name}")
                return None

        except Exception as e:
            logger.error(f"Error fetching funding rate for {symbol}: {e}")
            return None

    def fetch_open_interest(self, symbol: str) -> Optional[Dict]:
        """
        Fetch open interest for a futures symbol.
        Returns dict with 'openInterest', 'openInterestValue'
        """
        try:
            cache_key = symbol
            if cache_key in self._oi_cache:
                cached, cached_time = self._oi_cache[cache_key]
                if (datetime.now() - cached_time).total_seconds() < 60:
                    return cached

            if hasattr(self.exchange, 'fetch_open_interest'):
                oi = self.exchange.fetch_open_interest(symbol)
                result = {
                    'symbol': symbol,
                    'open_interest': oi.get('openInterest', 0),
                    'open_interest_value': oi.get('openInterestValue', 0),
                    'timestamp': oi.get('timestamp')
                }
                self._oi_cache[cache_key] = (result, datetime.now())
                return result
            else:
                logger.debug(f"Open interest not available for {self.exchange_name}")
                return None

        except Exception as e:
            logger.error(f"Error fetching open interest for {symbol}: {e}")
            return None

    def fetch_order_book(self, symbol: str, limit: int = 20) -> Optional[Dict]:
        """
        Fetch order book for a symbol.
        Returns dict with 'bids', 'asks', 'bid_volume', 'ask_volume'
        """
        try:
            orderbook = self.exchange.fetch_order_book(symbol, limit)

            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])

            bid_volume = sum(b[1] for b in bids) if bids else 0
            ask_volume = sum(a[1] for a in asks) if asks else 0

            return {
                'symbol': symbol,
                'bids': bids,
                'asks': asks,
                'bid_volume': bid_volume,
                'ask_volume': ask_volume,
                'spread': (asks[0][0] - bids[0][0]) if bids and asks else 0,
                'spread_percent': ((asks[0][0] - bids[0][0]) / bids[0][0] * 100) if bids and asks else 0,
                'imbalance': (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0
            }

        except Exception as e:
            logger.error(f"Error fetching order book for {symbol}: {e}")
            return None

    def fetch_recent_trades(self, symbol: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """Fetch recent trades for CVD calculation"""
        try:
            trades = self.exchange.fetch_trades(symbol, limit=limit)

            if not trades:
                return None

            df = pd.DataFrame(trades)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['side_multiplier'] = df['side'].apply(lambda x: 1 if x == 'buy' else -1)
            df['signed_volume'] = df['amount'] * df['side_multiplier']

            return df

        except Exception as e:
            logger.error(f"Error fetching trades for {symbol}: {e}")
            return None

    def fetch_ticker(self, symbol: str) -> Optional[Dict]:
        """Fetch current ticker data"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return {
                'symbol': symbol,
                'last': ticker.get('last'),
                'bid': ticker.get('bid'),
                'ask': ticker.get('ask'),
                'high': ticker.get('high'),
                'low': ticker.get('low'),
                'volume': ticker.get('baseVolume'),
                'quote_volume': ticker.get('quoteVolume'),
                'change_percent': ticker.get('percentage'),
                'vwap': ticker.get('vwap')
            }
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            return None

    def get_available_pairs(self, pairs: List[str]) -> Dict[str, bool]:
        """Check which pairs are available"""
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

    def get_market_info(self, symbol: str) -> Optional[Dict]:
        """Get market information for a symbol"""
        try:
            if symbol in self.exchange.markets:
                market = self.exchange.markets[symbol]
                return {
                    'symbol': symbol,
                    'base': market.get('base'),
                    'quote': market.get('quote'),
                    'type': market.get('type'),
                    'contract_size': market.get('contractSize'),
                    'tick_size': market.get('precision', {}).get('price'),
                    'min_amount': market.get('limits', {}).get('amount', {}).get('min'),
                    'maker_fee': market.get('maker'),
                    'taker_fee': market.get('taker')
                }
            return None
        except Exception as e:
            logger.error(f"Error getting market info for {symbol}: {e}")
            return None


# Alias for compatibility
EnhancedExchangeClient = DataClient
