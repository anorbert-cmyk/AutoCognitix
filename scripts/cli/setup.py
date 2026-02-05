#!/usr/bin/env python3
"""
Setup script for AutoCognitix CLI.

This allows the CLI to be installed as a pip package with an entry point.

Installation:
    pip install -e scripts/cli/

    # Or install globally
    pip install scripts/cli/

Usage after installation:
    autocognitix --help
    autocognitix diagnose P0171 --symptoms "Motor vibral"
    autocognitix dtc search "oxygen sensor"
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent.parent.parent / "docs" / "CLI.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

setup(
    name="autocognitix-cli",
    version="1.0.0",
    description="AutoCognitix Vehicle Diagnostic CLI Tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="AutoCognitix Team",
    author_email="info@autocognitix.com",
    url="https://github.com/autocognitix/autocognitix",
    license="MIT",

    # Package configuration
    py_modules=["autocognitix_cli"],
    packages=find_packages(),

    # Python version requirement
    python_requires=">=3.9",

    # Dependencies
    install_requires=[
        "click>=8.0.0",
        "rich>=13.0.0",
        "httpx>=0.24.0",
        "pyyaml>=6.0",
    ],

    # Optional dependencies for development
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
        ],
    },

    # Entry points for CLI commands
    entry_points={
        "console_scripts": [
            "autocognitix=autocognitix_cli:main",
        ],
    },

    # Classifiers for PyPI
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Utilities",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],

    # Keywords for search
    keywords=[
        "automotive",
        "diagnostics",
        "dtc",
        "obd",
        "vehicle",
        "cli",
        "hungarian",
    ],

    # Additional package data
    package_data={
        "": ["*.yaml", "*.json"],
    },

    # Include non-Python files
    include_package_data=True,
)
