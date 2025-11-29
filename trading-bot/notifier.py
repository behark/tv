import os
import logging
import requests
import time
from typing import Dict, Optional, List
from queue import Queue
from threading import Thread, Lock
from datetime import datetime

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2  # Base delay in seconds for exponential backoff

class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self._message_queue = Queue()
        self._queue_lock = Lock()
        self._last_message_time = None
        self._is_connected = False
        self._failed_count = 0
        self._success_count = 0

        if not bot_token or not chat_id:
            logger.error("Telegram bot token or chat ID not configured")
            raise ValueError("Telegram credentials not configured")

        # Test connection on init
        self._is_connected = self._test_connection()
        logger.info(f"Telegram notifier initialized (connected: {self._is_connected})")
    
    def _test_connection(self) -> bool:
        """Test if we can connect to Telegram API"""
        try:
            url = f"{self.base_url}/getMe"
            response = requests.get(url, timeout=10)
            return response.status_code == 200
        except Exception:
            return False

    def send_message(self, message: str, parse_mode: str = "HTML", priority: str = "normal") -> bool:
        """
        Send message with retry logic and exponential backoff.
        Priority: 'high' for TP/SL alerts, 'normal' for regular messages
        """
        for attempt in range(MAX_RETRIES):
            try:
                url = f"{self.base_url}/sendMessage"
                payload = {
                    "chat_id": self.chat_id,
                    "text": message,
                    "parse_mode": parse_mode
                }

                response = requests.post(url, json=payload, timeout=15)
                response.raise_for_status()

                self._is_connected = True
                self._success_count += 1
                self._last_message_time = datetime.now()
                logger.info(f"Message sent successfully to Telegram (attempt {attempt + 1})")
                return True

            except requests.exceptions.RequestException as e:
                self._failed_count += 1
                delay = RETRY_DELAY_BASE ** (attempt + 1)
                logger.warning(f"Failed to send message (attempt {attempt + 1}/{MAX_RETRIES}): {e}")

                if attempt < MAX_RETRIES - 1:
                    logger.info(f"Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"Failed to send message after {MAX_RETRIES} attempts")
                    self._is_connected = False
                    # Queue message for later if high priority
                    if priority == "high":
                        self._queue_message(message, parse_mode)
                    return False

            except Exception as e:
                logger.error(f"Unexpected error sending message: {e}")
                self._failed_count += 1
                return False

        return False

    def _queue_message(self, message: str, parse_mode: str = "HTML"):
        """Queue message for later delivery"""
        with self._queue_lock:
            self._message_queue.put((message, parse_mode, datetime.now()))
            logger.info(f"Message queued for later delivery (queue size: {self._message_queue.qsize()})")

    def flush_queue(self) -> int:
        """Attempt to send all queued messages"""
        sent_count = 0
        with self._queue_lock:
            while not self._message_queue.empty():
                message, parse_mode, queued_at = self._message_queue.get()
                # Add note about delayed delivery
                delayed_msg = f"{message}\n\n<i>⏰ Delayed delivery (queued at {queued_at.strftime('%H:%M:%S')})</i>"
                if self.send_message(delayed_msg, parse_mode):
                    sent_count += 1
                else:
                    # Put back in queue if still failing
                    self._message_queue.put((message, parse_mode, queued_at))
                    break
        return sent_count

    def get_stats(self) -> Dict:
        """Get notifier statistics"""
        return {
            'is_connected': self._is_connected,
            'success_count': self._success_count,
            'failed_count': self._failed_count,
            'queue_size': self._message_queue.qsize(),
            'last_message_time': self._last_message_time
        }
    
    def send_signal_alert(self, signal: Dict) -> bool:
        try:
            action = signal['action']
            symbol = signal['symbol']
            entry_price = signal['entry_price']
            stop_loss = signal['stop_loss']
            take_profit = signal['take_profit']
            score = signal['score']
            max_score = signal['max_score']
            rsi = signal['rsi']
            adx = signal['adx']
            
            confidence_pct = (score / max_score) * 100
            
            if action == 'LONG':
                emoji = "🟢"
            elif action == 'SHORT':
                emoji = "🔴"
            else:
                emoji = "📊"
            
            # Fix: Better handling of invalid R:R scenarios
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            if risk <= 0 or risk < 1e-10:
                logger.warning(f"Invalid risk calculation for {symbol}: risk={risk}")
                rr_ratio = 0
                rr_display = "Invalid"
            else:
                rr_ratio = reward / risk
                rr_display = f"1:{rr_ratio:.2f}"
            
            message = f"{emoji} <b>{action} SIGNAL - {symbol}</b>\n\n"
            message += f"💰 <b>Entry Price:</b> {entry_price:.6f}\n"
            message += f"🛑 <b>Stop Loss:</b> {stop_loss:.6f}\n"
            message += f"🎯 <b>Take Profit:</b> {take_profit:.6f}\n"
            message += f"📊 <b>Risk/Reward:</b> {rr_display}\n\n"
            message += f"⭐ <b>Confidence Score:</b> {score:.1f}/{max_score:.1f} ({confidence_pct:.0f}%)\n"
            message += f"📈 <b>RSI:</b> {rsi:.1f}\n"
            message += f"💪 <b>ADX:</b> {adx:.1f}\n"
            message += f"⏰ <b>Timeframe:</b> 15m\n"
            message += f"🔔 <b>Exchange:</b> MEXC Futures"
            
            return self.send_message(message)
            
        except Exception as e:
            logger.error(f"Error formatting signal alert: {e}")
            return False
    
    def send_startup_message(self, pairs: list, available_pairs: dict) -> bool:
        try:
            available_count = sum(1 for v in available_pairs.values() if v)
            total_count = len(pairs)
            
            message = "🤖 <b>Trading Bot Started</b>\n\n"
            message += f"📊 <b>Monitoring {available_count}/{total_count} pairs on MEXC Futures (15m)</b>\n\n"
            
            message += "<b>Available Pairs:</b>\n"
            for pair, available in available_pairs.items():
                if available:
                    message += f"✅ {pair}\n"
            
            unavailable = [pair for pair, available in available_pairs.items() if not available]
            if unavailable:
                message += f"\n<b>Unavailable Pairs:</b>\n"
                for pair in unavailable:
                    message += f"❌ {pair}\n"
            
            message += f"\n⚙️ <b>Strategy:</b> Confluence (EMA, RSI, MACD, ADX, Volume)\n"
            message += f"🎯 <b>Alert Mode:</b> Entry/Exit with TP/SL\n"
            message += f"✨ Ready to send trading signals!"
            
            return self.send_message(message)
            
        except Exception as e:
            logger.error(f"Error sending startup message: {e}")
            return False
    
    def send_error_message(self, error: str) -> bool:
        try:
            message = f"⚠️ <b>Trading Bot Error</b>\n\n{error}"
            return self.send_message(message)
        except Exception as e:
            logger.error(f"Error sending error message: {e}")
            return False
