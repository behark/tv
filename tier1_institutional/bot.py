#!/usr/bin/env python3
"""
Tier 1 - Institutional Trading Bot
Uses institutional-grade indicators: VWAP, CVD, Ichimoku, OBV, EMA Ribbon
"""

import sys
import os

# Add shared module path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))

from base_bot import BaseBot
from indicators import Tier1Indicators
from strategy import Tier1Strategy


class Tier1Bot(BaseBot):
    """
    Tier 1 Institutional Bot

    Focus: Institutional order flow and volume analysis
    Indicators: VWAP, CVD, Ichimoku Cloud, OBV, CMF, EMA Ribbon
    Best for: Identifying institutional buying/selling pressure
    """

    def _create_indicators(self):
        """Create Tier 1 indicator calculator."""
        return Tier1Indicators(self.config)

    def _create_strategy(self):
        """Create Tier 1 strategy."""
        return Tier1Strategy(self.config)


def main():
    """Main entry point."""
    # Get config path
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')

    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    # Create and run bot
    bot = Tier1Bot(config_path)
    bot.start()


if __name__ == "__main__":
    main()
