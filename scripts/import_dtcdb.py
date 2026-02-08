#!/usr/bin/env python3
"""
DTCDB (Diagnostic Trouble Code Database) Importer.

This script downloads and imports DTC codes from the todrobbins/dtcdb GitHub repository.

Source: https://github.com/todrobbins/dtcdb
Format: CSV (generic.csv)
License: Public Domain

Usage:
    python scripts/import_dtcdb.py              # Download, parse, and save
    python scripts/import_dtcdb.py --verbose    # With detailed logging

Output:
    data/dtc/dtcdb_codes.json
"""

import argparse
import json
import logging
import re
import sys
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "dtc"
OUTPUT_FILE = OUTPUT_DIR / "dtcdb_codes.json"

# GitHub raw URL
DTCDB_URL = "https://raw.githubusercontent.com/todrobbins/dtcdb/master/generic.csv"

# DTC code pattern
DTC_PATTERN = re.compile(r"^[PCBU][0-9A-F]{4}$", re.IGNORECASE)

# Category header pattern: "DTC Codes - P0100-P0199 – Fuel and Air Metering"
CATEGORY_HEADER_PATTERN = re.compile(
    r"^DTC Codes\s*[-–]\s*[PCBU]\d{4}-[PCBU]\d{4}\s*[-–]\s*(.+)$", re.IGNORECASE
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Category mappings based on DTC code prefix
CATEGORY_NAMES = {
    "P": "Powertrain",
    "C": "Chassis",
    "B": "Body",
    "U": "Network",
}

# Subcategory ranges for P codes
P_SUBCATEGORIES = {
    (0, 99): "Fuel and Air Metering",
    (100, 199): "Fuel and Air Metering",
    (200, 299): "Fuel and Air Metering (Injector Circuit)",
    (300, 399): "Ignition System or Misfire",
    (400, 499): "Auxiliary Emissions Controls",
    (500, 599): "Vehicle Speed and Idle Control",
    (600, 699): "Computer Output Circuit",
    (700, 899): "Transmission",
    (900, 999): "Transmission",
    (1000, 9999): "Manufacturer Specific",
}


@dataclass
class DTCCode:
    """DTC code data structure."""

    code: str
    description: str
    category: str
    subcategory: str


def get_category(code: str) -> str:
    """Get the main category from DTC code prefix."""
    if not code:
        return "Unknown"
    prefix = code[0].upper()
    return CATEGORY_NAMES.get(prefix, "Unknown")


def get_subcategory(code: str, parsed_subcategory: Optional[str] = None) -> str:
    """Get the subcategory based on DTC code range."""
    if parsed_subcategory:
        return parsed_subcategory

    if not code or len(code) < 5:
        return ""

    prefix = code[0].upper()

    # Only P codes have detailed subcategory ranges
    if prefix != "P":
        return CATEGORY_NAMES.get(prefix, "")

    try:
        # Extract numeric part (e.g., P0100 -> 100)
        num = int(code[1:5], 16) if code[1].upper() in "ABCDEF" else int(code[1:5])

        for (start, end), subcategory in P_SUBCATEGORIES.items():
            if start <= num <= end:
                return subcategory
    except ValueError:
        pass

    return "Powertrain"


def download_csv() -> str:
    """Download the generic.csv file from dtcdb repository."""
    logger.info(f"Downloading from {DTCDB_URL}")

    try:
        with urllib.request.urlopen(DTCDB_URL, timeout=30) as response:
            content = response.read().decode("utf-8")
            logger.info(f"Downloaded {len(content)} bytes")
            return content
    except urllib.error.URLError as e:
        logger.error(f"Failed to download: {e}")
        raise


def parse_csv(content: str) -> list[DTCCode]:
    """
    Parse the CSV content and extract DTC codes.

    The CSV format has:
    - Header row: "dtc, description"
    - Empty lines
    - Category headers: "DTC Codes - P0100-P0199 - Fuel and Air Metering"
    - Data rows: "P0100,Mass or Volume Air Flow Circuit Malfunction"
    """
    codes: list[DTCCode] = []
    current_subcategory = ""
    seen_codes: set[str] = set()

    lines = content.strip().split("\n")
    logger.info(f"Parsing {len(lines)} lines")

    for line in lines:
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Skip header row
        if line.lower().startswith("dtc,"):
            continue

        # Check for category header
        header_match = CATEGORY_HEADER_PATTERN.match(line)
        if header_match:
            current_subcategory = header_match.group(1).strip()
            logger.debug(f"Found subcategory: {current_subcategory}")
            continue

        # Try to parse as data row
        parts = line.split(",", 1)
        if len(parts) != 2:
            continue

        code = parts[0].strip().upper()
        description = parts[1].strip()

        # Validate DTC code format
        if not DTC_PATTERN.match(code):
            continue

        # Skip duplicates (there's a duplicate P0109 in the source)
        if code in seen_codes:
            logger.debug(f"Skipping duplicate: {code}")
            continue

        seen_codes.add(code)

        # Create DTC code entry
        dtc = DTCCode(
            code=code,
            description=description,
            category=get_category(code),
            subcategory=get_subcategory(code, current_subcategory),
        )
        codes.append(dtc)

    logger.info(f"Parsed {len(codes)} unique DTC codes")
    return codes


def save_json(codes: list[DTCCode], output_path: Path) -> None:
    """Save DTC codes to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "metadata": {
            "source": "https://github.com/todrobbins/dtcdb",
            "file": "generic.csv",
            "downloaded_at": datetime.utcnow().isoformat() + "Z",
            "total_codes": len(codes),
        },
        "codes": [asdict(c) for c in codes],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved to {output_path}")


def print_statistics(codes: list[DTCCode]) -> None:
    """Print statistics about the imported codes."""
    print("\n" + "=" * 60)
    print("DTCDB IMPORT STATISTICS")
    print("=" * 60)
    print(f"Total DTC codes: {len(codes)}")

    # Category distribution (P/C/B/U)
    category_counts: dict[str, int] = {}
    for code in codes:
        cat = code.category
        category_counts[cat] = category_counts.get(cat, 0) + 1

    print("\nCategory distribution (P/C/B/U):")
    for cat, count in sorted(category_counts.items()):
        prefix = {"Powertrain": "P", "Chassis": "C", "Body": "B", "Network": "U"}.get(
            cat, "?"
        )
        pct = (count / len(codes)) * 100
        print(f"  {prefix} ({cat}): {count} ({pct:.1f}%)")

    # Subcategory distribution
    subcategory_counts: dict[str, int] = {}
    for code in codes:
        subcat = code.subcategory or "Other"
        subcategory_counts[subcat] = subcategory_counts.get(subcat, 0) + 1

    print("\nSubcategory distribution:")
    for subcat, count in sorted(subcategory_counts.items(), key=lambda x: -x[1]):
        print(f"  {subcat}: {count}")

    print("=" * 60)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Import DTC codes from todrobbins/dtcdb GitHub repository"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=OUTPUT_FILE,
        help=f"Output file path (default: {OUTPUT_FILE})",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Download CSV
        csv_content = download_csv()

        # Parse CSV
        codes = parse_csv(csv_content)

        if not codes:
            logger.error("No DTC codes parsed. Check the source file format.")
            return 1

        # Save to JSON
        save_json(codes, args.output)

        # Print statistics
        print_statistics(codes)

        logger.info("Import completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Import failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
