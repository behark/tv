#!/usr/bin/env python3
"""
Trade Execution Module
Handles order placement, position management, and trade lifecycle.

IMPORTANT: This module executes REAL trades. Use with caution.
Always test with small amounts first.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import ccxt.async_support as ccxt

logger = logging.getLogger(__name__)


class OrderType(Enum):
    MARKET = 'market'
    LIMIT = 'limit'
    STOP_LOSS = 'stop_loss'
    TAKE_PROFIT = 'take_profit'


class OrderSide(Enum):
    BUY = 'buy'
    SELL = 'sell'


class PositionStatus(Enum):
    OPEN = 'open'
    CLOSED = 'closed'
    PARTIALLY_CLOSED = 'partially_closed'


@dataclass
class Order:
    """Represents an order."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    amount: float
    price: Optional[float]
    status: str
    filled: float = 0.0
    remaining: float = 0.0
    cost: float = 0.0
    fee: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Position:
    """Represents an open position."""
    position_id: str
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    amount: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    entry_order_id: Optional[str] = None
    sl_order_id: Optional[str] = None
    tp_order_id: Optional[str] = None
    status: PositionStatus = PositionStatus.OPEN
    entry_time: datetime = field(default_factory=datetime.now)
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl_percent: Optional[float] = None
    pnl_absolute: Optional[float] = None


class TradeExecutor:
    """
    Executes trades on exchanges.

    Supports:
    - Market and limit orders
    - Stop loss and take profit orders
    - Position tracking
    - Risk management
    """

    def __init__(self, exchange_id: str, api_key: str, api_secret: str,
                 config: dict = None, sandbox: bool = True):
        """
        Initialize trade executor.

        Args:
            exchange_id: Exchange identifier (e.g., 'mexc', 'binance')
            api_key: Exchange API key
            api_secret: Exchange API secret
            config: Additional configuration
            sandbox: Use testnet/sandbox mode (RECOMMENDED for testing)
        """
        self.config = config or {}
        self.sandbox = sandbox

        # Initialize exchange
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class({
            'apiKey': api_key,
            'secret': api_secret,
            'sandbox': sandbox,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',  # or 'future' for futures trading
            }
        })

        # Position tracking
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.position_counter = 0

        # Risk management settings
        self.max_position_size_percent = self.config.get('max_position_size_percent', 5.0)
        self.max_open_positions = self.config.get('max_open_positions', 5)
        self.use_stop_loss = self.config.get('use_stop_loss', True)
        self.use_take_profit = self.config.get('use_take_profit', True)

        logger.info(f"TradeExecutor initialized for {exchange_id} (sandbox={sandbox})")

    async def get_balance(self, currency: str = 'USDT') -> float:
        """Get available balance for a currency."""
        try:
            balance = await self.exchange.fetch_balance()
            return balance.get(currency, {}).get('free', 0.0)
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return 0.0

    async def get_current_price(self, symbol: str) -> float:
        """Get current market price for a symbol."""
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            return 0.0

    def calculate_position_size(self, balance: float, risk_percent: float,
                                 entry_price: float, stop_loss: float) -> float:
        """
        Calculate position size based on risk management.

        Args:
            balance: Available balance
            risk_percent: Percentage of balance to risk
            entry_price: Entry price
            stop_loss: Stop loss price

        Returns:
            Position size in base currency
        """
        risk_amount = balance * (risk_percent / 100)

        # Calculate distance to stop loss
        if entry_price > stop_loss:  # Long
            risk_per_unit = entry_price - stop_loss
        else:  # Short
            risk_per_unit = stop_loss - entry_price

        if risk_per_unit <= 0:
            return 0.0

        # Position size = risk amount / risk per unit
        position_size = risk_amount / risk_per_unit

        # Apply max position size limit
        max_position_value = balance * (self.max_position_size_percent / 100)
        max_position_size = max_position_value / entry_price

        return min(position_size, max_position_size)

    async def place_market_order(self, symbol: str, side: str, amount: float) -> Optional[Order]:
        """
        Place a market order.

        Args:
            symbol: Trading pair
            side: 'buy' or 'sell'
            amount: Amount in base currency

        Returns:
            Order object or None if failed
        """
        try:
            logger.info(f"Placing market {side} order: {amount} {symbol}")

            result = await self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=amount
            )

            order = Order(
                order_id=result['id'],
                symbol=symbol,
                side=OrderSide.BUY if side == 'buy' else OrderSide.SELL,
                order_type=OrderType.MARKET,
                amount=amount,
                price=result.get('average', result.get('price')),
                status=result['status'],
                filled=result.get('filled', 0),
                cost=result.get('cost', 0),
                fee=result.get('fee', {}).get('cost', 0)
            )

            self.orders[order.order_id] = order
            logger.info(f"Market order placed: {order.order_id}")

            return order

        except Exception as e:
            logger.error(f"Error placing market order: {e}")
            return None

    async def place_limit_order(self, symbol: str, side: str,
                                 amount: float, price: float) -> Optional[Order]:
        """Place a limit order."""
        try:
            logger.info(f"Placing limit {side} order: {amount} {symbol} @ {price}")

            result = await self.exchange.create_order(
                symbol=symbol,
                type='limit',
                side=side,
                amount=amount,
                price=price
            )

            order = Order(
                order_id=result['id'],
                symbol=symbol,
                side=OrderSide.BUY if side == 'buy' else OrderSide.SELL,
                order_type=OrderType.LIMIT,
                amount=amount,
                price=price,
                status=result['status'],
                filled=result.get('filled', 0)
            )

            self.orders[order.order_id] = order
            return order

        except Exception as e:
            logger.error(f"Error placing limit order: {e}")
            return None

    async def place_stop_loss_order(self, symbol: str, side: str,
                                     amount: float, stop_price: float) -> Optional[Order]:
        """
        Place a stop loss order.

        Note: Implementation varies by exchange. Some use stop-market,
        others use stop-limit orders.
        """
        try:
            logger.info(f"Placing stop loss: {amount} {symbol} @ {stop_price}")

            # Try stop-market first (preferred for SL)
            try:
                result = await self.exchange.create_order(
                    symbol=symbol,
                    type='stop_market',
                    side=side,
                    amount=amount,
                    params={'stopPrice': stop_price}
                )
            except:
                # Fallback to stop-limit
                result = await self.exchange.create_order(
                    symbol=symbol,
                    type='stop_limit',
                    side=side,
                    amount=amount,
                    price=stop_price * 0.995 if side == 'sell' else stop_price * 1.005,
                    params={'stopPrice': stop_price}
                )

            order = Order(
                order_id=result['id'],
                symbol=symbol,
                side=OrderSide.BUY if side == 'buy' else OrderSide.SELL,
                order_type=OrderType.STOP_LOSS,
                amount=amount,
                price=stop_price,
                status=result['status']
            )

            self.orders[order.order_id] = order
            return order

        except Exception as e:
            logger.error(f"Error placing stop loss: {e}")
            return None

    async def place_take_profit_order(self, symbol: str, side: str,
                                       amount: float, tp_price: float) -> Optional[Order]:
        """Place a take profit order."""
        try:
            logger.info(f"Placing take profit: {amount} {symbol} @ {tp_price}")

            # Take profit is typically a limit order
            result = await self.exchange.create_order(
                symbol=symbol,
                type='limit',
                side=side,
                amount=amount,
                price=tp_price
            )

            order = Order(
                order_id=result['id'],
                symbol=symbol,
                side=OrderSide.BUY if side == 'buy' else OrderSide.SELL,
                order_type=OrderType.TAKE_PROFIT,
                amount=amount,
                price=tp_price,
                status=result['status']
            )

            self.orders[order.order_id] = order
            return order

        except Exception as e:
            logger.error(f"Error placing take profit: {e}")
            return None

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an open order."""
        try:
            await self.exchange.cancel_order(order_id, symbol)
            if order_id in self.orders:
                self.orders[order_id].status = 'canceled'
            logger.info(f"Order canceled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Error canceling order {order_id}: {e}")
            return False

    async def open_position(self, signal: Dict) -> Optional[Position]:
        """
        Open a new position based on a signal.

        Args:
            signal: Signal dictionary with action, entry_price, stop_loss, take_profit

        Returns:
            Position object or None if failed
        """
        # Check if we can open more positions
        open_positions = [p for p in self.positions.values() if p.status == PositionStatus.OPEN]
        if len(open_positions) >= self.max_open_positions:
            logger.warning("Max open positions reached")
            return None

        symbol = signal['symbol']
        action = signal['action']  # LONG or SHORT
        entry_price = signal['entry_price']
        stop_loss = signal.get('stop_loss')
        take_profit = signal.get('take_profit')

        # Get balance and calculate position size
        balance = await self.get_balance('USDT')
        if balance <= 0:
            logger.error("Insufficient balance")
            return None

        risk_percent = self.config.get('risk_per_trade_percent', 1.0)

        if stop_loss:
            amount = self.calculate_position_size(balance, risk_percent, entry_price, stop_loss)
        else:
            # Fixed percentage of balance if no stop loss
            position_value = balance * (self.max_position_size_percent / 100)
            amount = position_value / entry_price

        if amount <= 0:
            logger.error("Calculated position size is zero")
            return None

        # Round to exchange precision
        try:
            markets = await self.exchange.load_markets()
            market = markets.get(symbol, {})
            precision = market.get('precision', {}).get('amount', 8)
            amount = round(amount, precision)
        except:
            amount = round(amount, 6)

        # Place entry order
        side = 'buy' if action == 'LONG' else 'sell'
        entry_order = await self.place_market_order(symbol, side, amount)

        if not entry_order:
            return None

        # Create position
        self.position_counter += 1
        position = Position(
            position_id=f"POS_{self.position_counter}",
            symbol=symbol,
            side='long' if action == 'LONG' else 'short',
            entry_price=entry_order.price or entry_price,
            amount=entry_order.filled or amount,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_order_id=entry_order.order_id
        )

        # Place SL/TP orders if configured
        exit_side = 'sell' if action == 'LONG' else 'buy'

        if self.use_stop_loss and stop_loss:
            sl_order = await self.place_stop_loss_order(
                symbol, exit_side, position.amount, stop_loss
            )
            if sl_order:
                position.sl_order_id = sl_order.order_id

        if self.use_take_profit and take_profit:
            tp_order = await self.place_take_profit_order(
                symbol, exit_side, position.amount, take_profit
            )
            if tp_order:
                position.tp_order_id = tp_order.order_id

        self.positions[position.position_id] = position
        logger.info(f"Position opened: {position.position_id} - {action} {amount} {symbol}")

        return position

    async def close_position(self, position_id: str,
                              reason: str = 'manual') -> Optional[Position]:
        """
        Close an open position.

        Args:
            position_id: Position ID to close
            reason: Reason for closing

        Returns:
            Updated position or None if failed
        """
        position = self.positions.get(position_id)
        if not position or position.status != PositionStatus.OPEN:
            logger.error(f"Position {position_id} not found or already closed")
            return None

        # Cancel any pending SL/TP orders
        if position.sl_order_id:
            await self.cancel_order(position.sl_order_id, position.symbol)
        if position.tp_order_id:
            await self.cancel_order(position.tp_order_id, position.symbol)

        # Place closing order
        side = 'sell' if position.side == 'long' else 'buy'
        close_order = await self.place_market_order(
            position.symbol, side, position.amount
        )

        if not close_order:
            return None

        # Update position
        position.status = PositionStatus.CLOSED
        position.exit_time = datetime.now()
        position.exit_price = close_order.price

        # Calculate P&L
        if position.side == 'long':
            position.pnl_percent = ((position.exit_price - position.entry_price) / position.entry_price) * 100
        else:
            position.pnl_percent = ((position.entry_price - position.exit_price) / position.entry_price) * 100

        position.pnl_absolute = (position.amount * position.entry_price) * (position.pnl_percent / 100)

        logger.info(f"Position closed: {position_id} - P&L: {position.pnl_percent:.2f}%")

        return position

    async def check_position_orders(self, position: Position) -> str:
        """
        Check if SL/TP orders have been filled.

        Returns:
            'open', 'tp_hit', 'sl_hit', or 'unknown'
        """
        if position.tp_order_id:
            try:
                order = await self.exchange.fetch_order(position.tp_order_id, position.symbol)
                if order['status'] == 'closed':
                    return 'tp_hit'
            except:
                pass

        if position.sl_order_id:
            try:
                order = await self.exchange.fetch_order(position.sl_order_id, position.symbol)
                if order['status'] == 'closed':
                    return 'sl_hit'
            except:
                pass

        return 'open'

    async def sync_positions(self):
        """
        Sync local position state with exchange.
        Check for filled SL/TP orders.
        """
        for position_id, position in list(self.positions.items()):
            if position.status != PositionStatus.OPEN:
                continue

            status = await self.check_position_orders(position)

            if status == 'tp_hit':
                position.status = PositionStatus.CLOSED
                position.exit_time = datetime.now()
                position.exit_price = position.take_profit

                if position.side == 'long':
                    position.pnl_percent = ((position.exit_price - position.entry_price) / position.entry_price) * 100
                else:
                    position.pnl_percent = ((position.entry_price - position.exit_price) / position.entry_price) * 100

                # Cancel SL order
                if position.sl_order_id:
                    await self.cancel_order(position.sl_order_id, position.symbol)

                logger.info(f"Position {position_id} TP hit: {position.pnl_percent:.2f}%")

            elif status == 'sl_hit':
                position.status = PositionStatus.CLOSED
                position.exit_time = datetime.now()
                position.exit_price = position.stop_loss

                if position.side == 'long':
                    position.pnl_percent = ((position.exit_price - position.entry_price) / position.entry_price) * 100
                else:
                    position.pnl_percent = ((position.entry_price - position.exit_price) / position.entry_price) * 100

                # Cancel TP order
                if position.tp_order_id:
                    await self.cancel_order(position.tp_order_id, position.symbol)

                logger.info(f"Position {position_id} SL hit: {position.pnl_percent:.2f}%")

    def get_open_positions(self) -> List[Position]:
        """Get all open positions."""
        return [p for p in self.positions.values() if p.status == PositionStatus.OPEN]

    def get_closed_positions(self) -> List[Position]:
        """Get all closed positions."""
        return [p for p in self.positions.values() if p.status == PositionStatus.CLOSED]

    def get_performance_summary(self) -> Dict:
        """Get trading performance summary."""
        closed = self.get_closed_positions()

        if not closed:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0
            }

        wins = [p for p in closed if p.pnl_percent > 0]
        losses = [p for p in closed if p.pnl_percent <= 0]

        return {
            'total_trades': len(closed),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': (len(wins) / len(closed)) * 100,
            'total_pnl_percent': sum(p.pnl_percent for p in closed),
            'total_pnl_absolute': sum(p.pnl_absolute or 0 for p in closed),
            'average_win': sum(p.pnl_percent for p in wins) / len(wins) if wins else 0,
            'average_loss': sum(p.pnl_percent for p in losses) / len(losses) if losses else 0
        }

    async def close(self):
        """Close exchange connection."""
        await self.exchange.close()
