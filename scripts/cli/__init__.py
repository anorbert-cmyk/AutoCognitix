"""
AutoCognitix CLI Tools
======================

Command-line interface tools for vehicle diagnostics.

Tools:
- autocognitix_cli.py: Main CLI with Click framework (recommended)
- diagtool.py: Legacy CLI with Typer framework

Installation:
    pip install -e scripts/cli/

Usage:
    autocognitix --help
    autocognitix diagnose P0171 --symptoms "Motor vibral"
    autocognitix dtc search "oxygen sensor"
"""

from pathlib import Path

__version__ = "1.0.0"
__all__ = ["diagtool", "autocognitix_cli"]

CLI_DIR = Path(__file__).parent
