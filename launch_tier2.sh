#!/bin/bash
# Launch Tier 2 - Advanced Technical Trading Bot

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "  Tier 2 - Advanced Technical Bot"
echo "  StochRSI, Fisher, FRAMA, Squeeze, HalfTrend"
echo "=========================================="

# Check for virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "Virtual environment activated"
fi

# Check environment variables
if [ -z "$TELEGRAM_BOT_TOKEN" ] || [ -z "$TELEGRAM_CHAT_ID" ]; then
    echo "Error: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set"
    echo "Export them or create a .env file"
    exit 1
fi

# Change to tier2 directory and run
cd tier2_advanced
python bot.py
