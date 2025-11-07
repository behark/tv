import os
import logging
import requests
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        
        if not bot_token or not chat_id:
            logger.error("Telegram bot token or chat ID not configured")
            raise ValueError("Telegram credentials not configured")
        
        logger.info("Telegram notifier initialized")
    
    def send_message(self, message: str, parse_mode: str = "HTML") -> bool:
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": parse_mode
            }
            
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info("Message sent successfully to Telegram")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send message to Telegram: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending message: {e}")
            return False
    
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
            
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            rr_ratio = reward / risk if risk > 0 else 0
            
            message = f"{emoji} <b>{action} SIGNAL - {symbol}</b>\n\n"
            message += f"💰 <b>Entry Price:</b> {entry_price:.6f}\n"
            message += f"🛑 <b>Stop Loss:</b> {stop_loss:.6f}\n"
            message += f"🎯 <b>Take Profit:</b> {take_profit:.6f}\n"
            message += f"📊 <b>Risk/Reward:</b> 1:{rr_ratio:.2f}\n\n"
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
