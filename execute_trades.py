#!/usr/bin/env python3
"""
Trade Execution Launcher
Run trading bots with live execution capabilities.

IMPORTANT: This executes REAL trades. Use sandbox mode for testing.

Usage:
    # Sandbox mode (recommended for testing)
    python execute_trades.py --tier 4 --sandbox

    # Live mode (REAL MONEY)
    python execute_trades.py --tier 4 --live

Environment Variables Required:
    EXCHANGE_API_KEY - Your exchange API key
    EXCHANGE_API_SECRET - Your exchange API secret
    TELEGRAM_BOT_TOKEN - Telegram bot token
    TELEGRAM_CHAT_ID - Telegram chat ID
"""

import argparse
import asyncio
import os
import sys
from getpass import getpass

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'shared'))


def get_tier_execution_bot(tier: int):
    """Get the execution bot class for a tier."""
    from execution_bot import ExecutionBot

    if tier == 1:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tier1_institutional'))
        from indicators import Tier1Indicators
        from strategy import Tier1Strategy

        class Tier1ExecutionBot(ExecutionBot):
            def _create_indicators(self):
                return Tier1Indicators(self.config)
            def _create_strategy(self):
                return Tier1Strategy(self.config)

        return Tier1ExecutionBot

    elif tier == 2:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tier2_advanced'))
        from indicators import Tier2Indicators
        from strategy import Tier2Strategy

        class Tier2ExecutionBot(ExecutionBot):
            def _create_indicators(self):
                return Tier2Indicators(self.config)
            def _create_strategy(self):
                return Tier2Strategy(self.config)

        return Tier2ExecutionBot

    elif tier == 3:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tier3_smart_money'))
        from indicators import Tier3Indicators
        from strategy import Tier3Strategy

        class Tier3ExecutionBot(ExecutionBot):
            def _create_indicators(self):
                return Tier3Indicators(self.config)
            def _create_strategy(self):
                return Tier3Strategy(self.config)

        return Tier3ExecutionBot

    elif tier == 4:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tier4_high_winrate'))
        from indicators import Tier4Indicators
        from strategy import Tier4Strategy

        class Tier4ExecutionBot(ExecutionBot):
            def _create_indicators(self):
                return Tier4Indicators(self.config)
            def _create_strategy(self):
                return Tier4Strategy(self.config)

        return Tier4ExecutionBot

    else:
        raise ValueError(f"Invalid tier: {tier}")


def get_config_path(tier: int) -> str:
    """Get config path for a tier."""
    paths = {
        1: 'tier1_institutional/config.yaml',
        2: 'tier2_advanced/config.yaml',
        3: 'tier3_smart_money/config.yaml',
        4: 'tier4_high_winrate/config.yaml'
    }
    return os.path.join(os.path.dirname(__file__), paths[tier])


def main():
    parser = argparse.ArgumentParser(
        description='Execute trades with trading bots',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Sandbox mode (testing):
    python execute_trades.py --tier 4 --sandbox

  Live mode (REAL MONEY):
    python execute_trades.py --tier 4 --live

Environment Variables:
  EXCHANGE_API_KEY     Your exchange API key
  EXCHANGE_API_SECRET  Your exchange API secret
  TELEGRAM_BOT_TOKEN   Telegram bot token
  TELEGRAM_CHAT_ID     Telegram chat ID
        """
    )

    parser.add_argument('--tier', type=int, required=True, choices=[1, 2, 3, 4],
                        help='Tier to run (1-4)')
    parser.add_argument('--sandbox', action='store_true',
                        help='Use sandbox/testnet mode (RECOMMENDED)')
    parser.add_argument('--live', action='store_true',
                        help='Use LIVE mode (REAL MONEY)')

    args = parser.parse_args()

    # Validate mode
    if args.live and args.sandbox:
        print("Error: Cannot use both --live and --sandbox")
        sys.exit(1)

    if not args.live and not args.sandbox:
        print("Error: Must specify --sandbox or --live")
        print("Use --sandbox for testing, --live for real trading")
        sys.exit(1)

    sandbox = args.sandbox

    # Get API credentials
    api_key = os.environ.get('EXCHANGE_API_KEY')
    api_secret = os.environ.get('EXCHANGE_API_SECRET')

    if not api_key:
        api_key = input("Enter Exchange API Key: ").strip()
    if not api_secret:
        api_secret = getpass("Enter Exchange API Secret: ").strip()

    if not api_key or not api_secret:
        print("Error: API credentials required")
        sys.exit(1)

    # Check Telegram credentials
    if not os.environ.get('TELEGRAM_BOT_TOKEN') or not os.environ.get('TELEGRAM_CHAT_ID'):
        print("Warning: Telegram credentials not set. Notifications will fail.")

    # Confirmation for live mode
    if not sandbox:
        print("\n" + "="*60)
        print("  ⚠️  WARNING: LIVE TRADING MODE  ⚠️")
        print("="*60)
        print("\nYou are about to start LIVE trading with REAL MONEY.")
        print("The bot will execute actual trades on your exchange account.")
        print("\nMake sure you understand the risks involved.")
        print("\n" + "="*60)

        confirm = input("\nType 'I UNDERSTAND' to continue: ")
        if confirm != 'I UNDERSTAND':
            print("Aborted.")
            sys.exit(0)

    # Get bot class and config
    BotClass = get_tier_execution_bot(args.tier)
    config_path = get_config_path(args.tier)

    tier_names = {
        1: "Institutional",
        2: "Advanced Technical",
        3: "Smart Money",
        4: "High Win Rate"
    }

    print("\n" + "="*60)
    mode = "🧪 SANDBOX" if sandbox else "💰 LIVE"
    print(f"  Starting Tier {args.tier} - {tier_names[args.tier]}")
    print(f"  Mode: {mode}")
    print("="*60 + "\n")

    # Create and run bot
    try:
        bot = BotClass(
            config_path=config_path,
            api_key=api_key,
            api_secret=api_secret,
            sandbox=sandbox
        )

        asyncio.run(bot.run())

    except KeyboardInterrupt:
        print("\nShutdown requested...")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
