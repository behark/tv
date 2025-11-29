#!/usr/bin/env python3
"""
Backtest CLI Tool
Run backtests on all trading tiers against historical data.

Usage:
    python backtest.py --tier 1 --symbol BTC/USDT --days 30
    python backtest.py --all --symbol ETH/USDT --days 90
    python backtest.py --tier 4 --symbol SOL/USDT --start 2024-01-01 --end 2024-06-01
"""

import argparse
import asyncio
import sys
import os
from datetime import datetime, timedelta
from typing import List

# Add shared module path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'shared'))

from backtester import BacktestEngine, BacktestReporter, BacktestResult
from data_client import EnhancedExchangeClient


def get_tier_components(tier: int):
    """Get indicator and strategy classes for a tier."""
    if tier == 1:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tier1_institutional'))
        from indicators import Tier1Indicators
        from strategy import Tier1Strategy
        return Tier1Indicators, Tier1Strategy

    elif tier == 2:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tier2_advanced'))
        from indicators import Tier2Indicators
        from strategy import Tier2Strategy
        return Tier2Indicators, Tier2Strategy

    elif tier == 3:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tier3_smart_money'))
        from indicators import Tier3Indicators
        from strategy import Tier3Strategy
        return Tier3Indicators, Tier3Strategy

    elif tier == 4:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tier4_high_winrate'))
        from indicators import Tier4Indicators
        from strategy import Tier4Strategy
        return Tier4Indicators, Tier4Strategy

    else:
        raise ValueError(f"Invalid tier: {tier}. Must be 1-4.")


def load_tier_config(tier: int) -> dict:
    """Load configuration for a tier."""
    import yaml

    config_paths = {
        1: 'tier1_institutional/config.yaml',
        2: 'tier2_advanced/config.yaml',
        3: 'tier3_smart_money/config.yaml',
        4: 'tier4_high_winrate/config.yaml'
    }

    config_path = os.path.join(os.path.dirname(__file__), config_paths[tier])

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


async def fetch_historical_data(symbol: str, timeframe: str,
                                 start_date: datetime, end_date: datetime) -> 'pd.DataFrame':
    """Fetch historical OHLCV data from exchange."""
    import pandas as pd

    client = EnhancedExchangeClient(
        exchange_id='mexc',
        config={'enableRateLimit': True, 'timeout': 30000}
    )

    await client.load_markets()

    formatted_symbol = client.format_symbol(symbol)

    # Calculate number of candles needed
    timeframe_minutes = {
        '1m': 1, '5m': 5, '15m': 15, '30m': 30,
        '1h': 60, '4h': 240, '1d': 1440
    }

    minutes = timeframe_minutes.get(timeframe, 15)
    total_minutes = (end_date - start_date).total_seconds() / 60
    candles_needed = int(total_minutes / minutes)

    print(f"Fetching {candles_needed} candles for {symbol}...")

    # Fetch in chunks (exchange limit is usually 1000)
    all_data = []
    since = int(start_date.timestamp() * 1000)
    end_ts = int(end_date.timestamp() * 1000)

    while since < end_ts:
        try:
            ohlcv = await client.exchange.fetch_ohlcv(
                formatted_symbol,
                timeframe=timeframe,
                since=since,
                limit=1000
            )

            if not ohlcv:
                break

            all_data.extend(ohlcv)
            since = ohlcv[-1][0] + 1

            # Progress indicator
            progress = (since - int(start_date.timestamp() * 1000)) / (end_ts - int(start_date.timestamp() * 1000)) * 100
            print(f"  Progress: {progress:.1f}%", end='\r')

            await asyncio.sleep(0.1)  # Rate limiting

        except Exception as e:
            print(f"\nError fetching data: {e}")
            break

    print(f"\nFetched {len(all_data)} candles")

    if not all_data:
        return None

    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')

    return df


async def run_single_backtest(tier: int, symbol: str, timeframe: str,
                               start_date: datetime, end_date: datetime,
                               config: dict = None) -> BacktestResult:
    """Run backtest for a single tier."""
    print(f"\n{'='*60}")
    print(f"Running backtest for Tier {tier}")
    print(f"Symbol: {symbol}, Timeframe: {timeframe}")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print('='*60)

    # Load tier config
    tier_config = load_tier_config(tier)

    # Override with any provided config
    if config:
        tier_config.update(config)

    # Get tier components
    IndicatorClass, StrategyClass = get_tier_components(tier)

    # Create instances
    indicators = IndicatorClass(tier_config)
    strategy = StrategyClass(tier_config)

    # Fetch historical data
    df = await fetch_historical_data(symbol, timeframe, start_date, end_date)

    if df is None or len(df) < 100:
        print(f"Insufficient data for Tier {tier} backtest")
        return None

    # Create backtest engine
    backtest_config = {
        'initial_capital': 10000,
        'position_size_percent': 2.0,
        'max_open_trades': 3,
        'commission_percent': 0.1,
        'slippage_percent': 0.05,
        'timeframe': timeframe
    }

    engine = BacktestEngine(backtest_config)

    # Run backtest
    print("Running backtest simulation...")
    result = engine.run(df, indicators, strategy, symbol)

    return result


async def run_all_tiers_backtest(symbol: str, timeframe: str,
                                  start_date: datetime, end_date: datetime) -> List[BacktestResult]:
    """Run backtest for all tiers."""
    results = []

    for tier in [1, 2, 3, 4]:
        try:
            result = await run_single_backtest(tier, symbol, timeframe, start_date, end_date)
            if result:
                results.append(result)
                BacktestReporter.print_summary(result)
        except Exception as e:
            print(f"Error running Tier {tier} backtest: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Backtest trading strategies')

    parser.add_argument('--tier', type=int, choices=[1, 2, 3, 4],
                        help='Tier to backtest (1-4)')
    parser.add_argument('--all', action='store_true',
                        help='Run backtest on all tiers')
    parser.add_argument('--symbol', type=str, default='BTC/USDT',
                        help='Trading pair symbol (default: BTC/USDT)')
    parser.add_argument('--timeframe', type=str, default='15m',
                        help='Candle timeframe (default: 15m)')
    parser.add_argument('--days', type=int, default=30,
                        help='Number of days to backtest (default: 30)')
    parser.add_argument('--start', type=str,
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str,
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str,
                        help='Output file for results (JSON)')

    args = parser.parse_args()

    # Validate arguments
    if not args.tier and not args.all:
        print("Error: Must specify --tier or --all")
        parser.print_help()
        sys.exit(1)

    # Calculate date range
    if args.start and args.end:
        start_date = datetime.strptime(args.start, '%Y-%m-%d')
        end_date = datetime.strptime(args.end, '%Y-%m-%d')
    else:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)

    print("\n" + "="*60)
    print("  BACKTEST RUNNER")
    print("="*60)
    print(f"\nSymbol: {args.symbol}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Period: {start_date.date()} to {end_date.date()}")

    if args.all:
        print("Mode: All Tiers")
    else:
        print(f"Mode: Tier {args.tier}")

    # Run backtests
    if args.all:
        results = asyncio.run(run_all_tiers_backtest(
            args.symbol, args.timeframe, start_date, end_date
        ))

        if results:
            BacktestReporter.compare_results(results)

            if args.output:
                import json
                data = [BacktestReporter.to_dict(r) for r in results]
                with open(args.output, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"\nResults saved to {args.output}")

    else:
        result = asyncio.run(run_single_backtest(
            args.tier, args.symbol, args.timeframe, start_date, end_date
        ))

        if result:
            BacktestReporter.print_summary(result)

            if args.output:
                BacktestReporter.save_json(result, args.output)


if __name__ == '__main__':
    main()
