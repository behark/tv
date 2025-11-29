#!/usr/bin/env python3
"""
Quick Standalone Backtest
Simplified backtest that works without pandas_ta dependency.
Tests the core mean reversion / confluence concepts.
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class Trade:
    entry_time: datetime
    entry_price: float
    action: str  # LONG or SHORT
    stop_loss: float
    take_profit: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl_percent: Optional[float] = None


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std: float = 2.0):
    """Calculate Bollinger Bands."""
    middle = prices.rolling(period).mean()
    std_dev = prices.rolling(period).std()
    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)
    return upper, middle, lower


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate ATR."""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """Calculate EMA."""
    return prices.ewm(span=period, adjust=False).mean()


async def fetch_data(symbol: str, days: int = 30) -> pd.DataFrame:
    """Fetch historical data."""
    import ccxt.async_support as ccxt_async

    print(f"Fetching {days} days of data for {symbol}...")

    exchange = ccxt_async.mexc({'enableRateLimit': True, 'timeout': 30000})
    await exchange.load_markets()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    all_data = []
    since = int(start_date.timestamp() * 1000)
    end_ts = int(end_date.timestamp() * 1000)

    while since < end_ts:
        try:
            ohlcv = await exchange.fetch_ohlcv(symbol, '15m', since=since, limit=1000)
            if not ohlcv:
                break
            all_data.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            await asyncio.sleep(0.2)
        except Exception as e:
            print(f"Error: {e}")
            break

    await exchange.close()

    if not all_data:
        return None

    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')

    print(f"Fetched {len(df)} candles")
    return df


def run_strategy(df: pd.DataFrame, strategy_name: str) -> List[Trade]:
    """Run a strategy and return trades."""
    trades = []

    # Calculate indicators
    df['rsi'] = calculate_rsi(df['close'])
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = calculate_bollinger_bands(df['close'])
    df['atr'] = calculate_atr(df['high'], df['low'], df['close'])
    df['ema_21'] = calculate_ema(df['close'], 21)
    df['ema_50'] = calculate_ema(df['close'], 50)
    df['volume_sma'] = df['volume'].rolling(20).mean()

    # Strategy parameters based on tier
    if strategy_name == 'tier1':
        # Institutional: Trend following with volume
        min_score = 3
    elif strategy_name == 'tier2':
        # Advanced: Momentum based
        min_score = 3
    elif strategy_name == 'tier3':
        # SMC: Structure based
        min_score = 4
    else:  # tier4
        # High win rate: Mean reversion
        min_score = 5

    open_trade = None

    for i in range(50, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]

        # Check open trade for exit
        if open_trade:
            # Check stop loss
            if open_trade.action == 'LONG':
                if current['low'] <= open_trade.stop_loss:
                    open_trade.exit_time = df.index[i]
                    open_trade.exit_price = open_trade.stop_loss
                    open_trade.exit_reason = 'SL_HIT'
                    open_trade.pnl_percent = ((open_trade.exit_price - open_trade.entry_price) / open_trade.entry_price) * 100
                    trades.append(open_trade)
                    open_trade = None
                    continue
                elif current['high'] >= open_trade.take_profit:
                    open_trade.exit_time = df.index[i]
                    open_trade.exit_price = open_trade.take_profit
                    open_trade.exit_reason = 'TP_HIT'
                    open_trade.pnl_percent = ((open_trade.exit_price - open_trade.entry_price) / open_trade.entry_price) * 100
                    trades.append(open_trade)
                    open_trade = None
                    continue
            else:  # SHORT
                if current['high'] >= open_trade.stop_loss:
                    open_trade.exit_time = df.index[i]
                    open_trade.exit_price = open_trade.stop_loss
                    open_trade.exit_reason = 'SL_HIT'
                    open_trade.pnl_percent = ((open_trade.entry_price - open_trade.exit_price) / open_trade.entry_price) * 100
                    trades.append(open_trade)
                    open_trade = None
                    continue
                elif current['low'] <= open_trade.take_profit:
                    open_trade.exit_time = df.index[i]
                    open_trade.exit_price = open_trade.take_profit
                    open_trade.exit_reason = 'TP_HIT'
                    open_trade.pnl_percent = ((open_trade.entry_price - open_trade.exit_price) / open_trade.entry_price) * 100
                    trades.append(open_trade)
                    open_trade = None
                    continue

            continue  # Skip signal check if we have open trade

        # Calculate confluence score
        bull_score = 0
        bear_score = 0

        # RSI
        if current['rsi'] < 30:
            bull_score += 2
        if current['rsi'] < 20:
            bull_score += 1
        if current['rsi'] > 70:
            bear_score += 2
        if current['rsi'] > 80:
            bear_score += 1

        # Bollinger Bands
        if current['close'] <= current['bb_lower']:
            bull_score += 2
        if current['close'] >= current['bb_upper']:
            bear_score += 2

        # Volume spike
        if current['volume'] > current['volume_sma'] * 1.5:
            bull_score += 1
            bear_score += 1

        # EMA trend
        if current['ema_21'] > current['ema_50']:
            bull_score += 1
        else:
            bear_score += 1

        # Generate signals
        atr = current['atr']

        if bull_score >= min_score and bull_score > bear_score:
            entry = current['close']
            sl = entry - (atr * 1.5)
            tp = entry + (atr * 2.0)
            open_trade = Trade(
                entry_time=df.index[i],
                entry_price=entry,
                action='LONG',
                stop_loss=sl,
                take_profit=tp
            )

        elif bear_score >= min_score and bear_score > bull_score:
            entry = current['close']
            sl = entry + (atr * 1.5)
            tp = entry - (atr * 2.0)
            open_trade = Trade(
                entry_time=df.index[i],
                entry_price=entry,
                action='SHORT',
                stop_loss=sl,
                take_profit=tp
            )

    # Close any remaining open trade
    if open_trade:
        open_trade.exit_time = df.index[-1]
        open_trade.exit_price = df.iloc[-1]['close']
        open_trade.exit_reason = 'END'
        if open_trade.action == 'LONG':
            open_trade.pnl_percent = ((open_trade.exit_price - open_trade.entry_price) / open_trade.entry_price) * 100
        else:
            open_trade.pnl_percent = ((open_trade.entry_price - open_trade.exit_price) / open_trade.entry_price) * 100
        trades.append(open_trade)

    return trades


def print_results(strategy_name: str, trades: List[Trade]):
    """Print backtest results."""
    if not trades:
        print(f"\n{strategy_name}: No trades generated")
        return

    wins = [t for t in trades if t.pnl_percent > 0]
    losses = [t for t in trades if t.pnl_percent <= 0]
    tp_hits = [t for t in trades if t.exit_reason == 'TP_HIT']
    sl_hits = [t for t in trades if t.exit_reason == 'SL_HIT']

    total_pnl = sum(t.pnl_percent for t in trades)
    win_rate = (len(wins) / len(trades)) * 100 if trades else 0
    avg_win = np.mean([t.pnl_percent for t in wins]) if wins else 0
    avg_loss = np.mean([t.pnl_percent for t in losses]) if losses else 0

    print(f"\n{'='*50}")
    print(f"  {strategy_name.upper()} RESULTS")
    print(f"{'='*50}")
    print(f"Total Trades: {len(trades)}")
    print(f"Winning: {len(wins)} | Losing: {len(losses)}")
    print(f"TP Hits: {len(tp_hits)} | SL Hits: {len(sl_hits)}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Total P&L: {total_pnl:.2f}%")
    print(f"Avg Win: {avg_win:.2f}% | Avg Loss: {avg_loss:.2f}%")

    if wins and losses:
        profit_factor = abs(sum(t.pnl_percent for t in wins) / sum(t.pnl_percent for t in losses))
        print(f"Profit Factor: {profit_factor:.2f}")

    return {
        'name': strategy_name,
        'trades': len(trades),
        'win_rate': win_rate,
        'total_pnl': total_pnl
    }


async def main():
    print("\n" + "="*60)
    print("  QUICK BACKTEST - All Tiers")
    print("="*60)

    # Fetch data
    df = await fetch_data('BTC/USDT', days=14)

    if df is None or len(df) < 100:
        print("Failed to fetch sufficient data")
        return

    results = []

    # Test all strategies
    for tier in ['tier1', 'tier2', 'tier3', 'tier4']:
        trades = run_strategy(df.copy(), tier)
        result = print_results(tier, trades)
        if result:
            results.append(result)

    # Comparison
    print("\n" + "="*60)
    print("  COMPARISON SUMMARY")
    print("="*60)
    print(f"\n{'Strategy':<15} {'Trades':>8} {'Win Rate':>12} {'Total P&L':>12}")
    print("-" * 50)

    for r in results:
        print(f"{r['name']:<15} {r['trades']:>8} {r['win_rate']:>11.1f}% {r['total_pnl']:>11.2f}%")

    if results:
        best = max(results, key=lambda x: x['win_rate'])
        print(f"\nBest Win Rate: {best['name']} ({best['win_rate']:.1f}%)")

        best_pnl = max(results, key=lambda x: x['total_pnl'])
        print(f"Best P&L: {best_pnl['name']} ({best_pnl['total_pnl']:.2f}%)")


if __name__ == '__main__':
    asyncio.run(main())
