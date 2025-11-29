#!/usr/bin/env python3
"""
Tier 3 - Smart Money Concepts Trading Bot
Uses ICT-style indicators: Order Blocks, FVG, Liquidity Zones, BOS/CHoCH,
Premium/Discount Zones
"""

import sys
import os

# Add shared module path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))

from base_bot import BaseBot
from indicators import Tier3Indicators
from strategy import Tier3Strategy


class Tier3Bot(BaseBot):
    """
    Tier 3 Smart Money Concepts Bot

    Focus: ICT-style institutional trading concepts
    Indicators: Order Blocks, Fair Value Gaps (FVG), Liquidity Zones,
                Market Structure (BOS/CHoCH), Premium/Discount Zones
    Best for: Trading around institutional order flow and liquidity
    """

    def _create_indicators(self):
        """Create Tier 3 indicator calculator."""
        return Tier3Indicators(self.config)

    def _create_strategy(self):
        """Create Tier 3 strategy."""
        return Tier3Strategy(self.config)


def main():
    """Main entry point."""
    # Get config path
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')

    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    # Create and run bot
    bot = Tier3Bot(config_path)
    bot.start()


if __name__ == "__main__":
    main()
