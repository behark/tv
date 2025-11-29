#!/usr/bin/env python3
"""
Tier 2 - Advanced Technical Trading Bot
Uses advanced oscillators and volatility indicators: Stochastic RSI, Fisher Transform,
FRAMA, Waddah Attar Explosion, Squeeze Momentum, Half Trend
"""

import sys
import os

# Add shared module path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))

from base_bot import BaseBot
from indicators import Tier2Indicators
from strategy import Tier2Strategy


class Tier2Bot(BaseBot):
    """
    Tier 2 Advanced Technical Bot

    Focus: Momentum and volatility analysis
    Indicators: Stochastic RSI, Fisher Transform, FRAMA, Waddah Attar,
                Squeeze Momentum, Half Trend, Keltner Channels
    Best for: Catching momentum shifts and squeeze breakouts
    """

    def _create_indicators(self):
        """Create Tier 2 indicator calculator."""
        return Tier2Indicators(self.config)

    def _create_strategy(self):
        """Create Tier 2 strategy."""
        return Tier2Strategy(self.config)


def main():
    """Main entry point."""
    # Get config path
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')

    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    # Create and run bot
    bot = Tier2Bot(config_path)
    bot.start()


if __name__ == "__main__":
    main()
