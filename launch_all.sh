#!/bin/bash
# Launch All Trading Bots (runs in separate terminal windows/tmux panes)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "  Multi-Tier Trading Bot Launcher"
echo "  Tiers 1-4: All Strategies"
echo "=========================================="

# Check environment variables
if [ -z "$TELEGRAM_BOT_TOKEN" ] || [ -z "$TELEGRAM_CHAT_ID" ]; then
    echo "Error: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set"
    exit 1
fi

# Check if tmux is available
if command -v tmux &> /dev/null; then
    echo "Using tmux to launch all bots..."

    # Create new tmux session
    tmux new-session -d -s trading_bots -n tier1

    # Launch Tier 1 in first window
    tmux send-keys -t trading_bots:tier1 "cd $SCRIPT_DIR && ./launch_tier1.sh" C-m

    # Create window for Tier 2
    tmux new-window -t trading_bots -n tier2
    tmux send-keys -t trading_bots:tier2 "cd $SCRIPT_DIR && ./launch_tier2.sh" C-m

    # Create window for Tier 3
    tmux new-window -t trading_bots -n tier3
    tmux send-keys -t trading_bots:tier3 "cd $SCRIPT_DIR && ./launch_tier3.sh" C-m

    # Create window for Tier 4
    tmux new-window -t trading_bots -n tier4
    tmux send-keys -t trading_bots:tier4 "cd $SCRIPT_DIR && ./launch_tier4.sh" C-m

    echo ""
    echo "All 4 bots launched in tmux session 'trading_bots'"
    echo "Attach with: tmux attach -t trading_bots"
    echo "Switch windows with: Ctrl+B then 0/1/2/3"
    echo ""
    echo "Tiers:"
    echo "  0: Tier 1 - Institutional (VWAP, CVD, Ichimoku)"
    echo "  1: Tier 2 - Advanced Technical (Squeeze, Fisher)"
    echo "  2: Tier 3 - Smart Money (Order Blocks, FVG)"
    echo "  3: Tier 4 - High Win Rate (Mean Reversion)"
    echo ""

    # Attach to session
    tmux attach -t trading_bots

else
    echo "tmux not found. Running bots in background..."
    echo "Install tmux for better multi-bot management: apt install tmux"
    echo ""

    # Run in background with nohup
    nohup ./launch_tier1.sh > tier1.out 2>&1 &
    echo "Tier 1 started (PID: $!)"

    nohup ./launch_tier2.sh > tier2.out 2>&1 &
    echo "Tier 2 started (PID: $!)"

    nohup ./launch_tier3.sh > tier3.out 2>&1 &
    echo "Tier 3 started (PID: $!)"

    nohup ./launch_tier4.sh > tier4.out 2>&1 &
    echo "Tier 4 started (PID: $!)"

    echo ""
    echo "All 4 bots running in background"
    echo "Check logs: tail -f tier1.out tier2.out tier3.out tier4.out"
fi
