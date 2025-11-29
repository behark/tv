#!/usr/bin/env python3
"""
Tier 4 - High Win Rate Trading Bot
Multi-confluence mean reversion targeting 75-85% accuracy.
"""

import sys
import os

# Add shared module path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))

from base_bot import BaseBot
from indicators import Tier4Indicators
from strategy import Tier4Strategy


class Tier4Bot(BaseBot):
    """
    Tier 4 High Win Rate Bot

    Focus: Maximum probability setups through multi-confluence
    Core Strategy: Mean reversion with multiple confirmations

    Entry Requirements:
    - RSI at extreme (oversold < 30 or overbought > 70)
    - Price at Bollinger Band extreme
    - Stochastic RSI confirmation
    - Volume spike or climax
    - Near support/resistance level
    - Optional: RSI divergence (strong bonus)

    Target: 75-85% win rate with 1.5:1 R:R

    Best for: Traders who prefer high win rate over high R:R
    """

    def _create_indicators(self):
        """Create Tier 4 indicator calculator."""
        return Tier4Indicators(self.config)

    def _create_strategy(self):
        """Create Tier 4 strategy."""
        return Tier4Strategy(self.config)


def main():
    """Main entry point."""
    # Get config path
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')

    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    # Create and run bot
    bot = Tier4Bot(config_path)
    bot.start()


if __name__ == "__main__":
    main()
