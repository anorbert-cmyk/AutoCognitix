#!/usr/bin/env python3
"""
Master DTC Database Scraper.

Runs all individual scrapers and merges results into all_codes_merged.json.

Usage:
    python scripts/scrape_all_dtc_sources.py              # Run all scrapers
    python scripts/scrape_all_dtc_sources.py --limit 50   # Limit codes per source
    python scripts/scrape_all_dtc_sources.py --sources obd-codes,dtcbase  # Specific sources
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Data paths
DATA_DIR = PROJECT_ROOT / "data" / "dtc_codes"
MASTER_FILE = DATA_DIR / "all_codes_merged.json"

# Available scrapers
SCRAPERS = {
    "obd-codes": "scrape_obd_codes",
    "dtcbase": "scrape_dtcbase",
    "troublecodes": "scrape_troublecodes",
    "autocodes": "scrape_autocodes",
    "bba-reman": "scrape_bbareman",
    "klavkarr": "scrape_klavkarr",
}


async def run_scraper(
    scraper_name: str,
    module_name: str,
    limit: Optional[int] = None,
) -> Tuple[str, List[Dict[str, Any]], Optional[str]]:
    """
    Run a single scraper module.

    Args:
        scraper_name: Human-readable name of the scraper.
        module_name: Python module name.
        limit: Optional limit on codes to scrape.

    Returns:
        Tuple of (scraper_name, codes, error_message).
    """
    try:
        logger.info(f"Starting scraper: {scraper_name}")

        # Import the module dynamically
        module = __import__(f"scripts.{module_name}", fromlist=[module_name])

        # Run the scraper
        if hasattr(module, "scrape_all_codes"):
            raw_codes = await module.scrape_all_codes(limit=limit)
        elif hasattr(module, "scrape_all_ranges"):
            raw_codes = await module.scrape_all_ranges()
        else:
            logger.error(f"Scraper {scraper_name} has no scrape function")
            return scraper_name, [], "No scrape function found"

        # Normalize if available
        if hasattr(module, "normalize_codes"):
            codes = module.normalize_codes(raw_codes)
        else:
            codes = raw_codes

        # Save to cache if available
        if hasattr(module, "save_to_cache"):
            module.save_to_cache(codes)

        logger.info(f"Completed {scraper_name}: {len(codes)} codes")
        return scraper_name, codes, None

    except Exception as e:
        logger.error(f"Error running {scraper_name}: {e}")
        return scraper_name, [], str(e)


def merge_all_codes(all_results: List[Tuple[str, List[Dict[str, Any]], Optional[str]]]) -> Dict[str, Any]:
    """
    Merge codes from all scrapers into master file.

    Args:
        all_results: List of (scraper_name, codes, error) tuples.

    Returns:
        Master data dictionary.
    """
    # Load existing master if it exists
    if MASTER_FILE.exists():
        with open(MASTER_FILE, "r", encoding="utf-8") as f:
            master_data = json.load(f)
        master_codes = {c["code"]: c for c in master_data.get("codes", [])}
        logger.info(f"Loaded {len(master_codes)} existing codes from master")
    else:
        master_data = {
            "metadata": {},
            "codes": [],
        }
        master_codes = {}

    new_count = 0
    updated_count = 0

    # Process results from each scraper
    for scraper_name, codes, error in all_results:
        if error:
            logger.warning(f"Skipping {scraper_name} due to error: {error}")
            continue

        source_name = scraper_name.replace("-", "_")

        for code_data in codes:
            code = code_data["code"]

            if code in master_codes:
                existing = master_codes[code]

                # Update sources list
                if "sources" not in existing:
                    existing["sources"] = []
                # Handle both source formats
                new_source = code_data.get("sources", [code_data.get("source", source_name)])
                if isinstance(new_source, str):
                    new_source = [new_source]
                for src in new_source:
                    if src and src not in existing["sources"]:
                        existing["sources"].append(src)

                # Update description if new one is better
                new_desc = code_data.get("description_en", "")
                existing_desc = existing.get("description_en", "")
                if len(new_desc) > len(existing_desc):
                    existing["description_en"] = new_desc
                    updated_count += 1

                # Merge list fields
                for field in ["symptoms", "possible_causes", "diagnostic_steps", "related_codes"]:
                    if code_data.get(field):
                        existing_items = set(existing.get(field, []))
                        for item in code_data[field]:
                            if item and item not in existing_items:
                                existing.setdefault(field, []).append(item)
            else:
                # Ensure sources field exists
                if "sources" not in code_data:
                    code_data["sources"] = [code_data.get("source", source_name)]
                if "source" in code_data:
                    del code_data["source"]

                master_codes[code] = code_data
                new_count += 1

    # Build final data
    master_data["codes"] = sorted(master_codes.values(), key=lambda x: x["code"])
    master_data["metadata"] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_codes": len(master_data["codes"]),
        "translated": sum(1 for c in master_data["codes"] if c.get("description_hu")),
        "sources": list(SCRAPERS.keys()),
        "new_codes_added": new_count,
        "codes_updated": updated_count,
    }

    return master_data


def print_summary(
    all_results: List[Tuple[str, List[Dict[str, Any]], Optional[str]]],
    master_data: Dict[str, Any],
) -> None:
    """Print summary of scraping results."""
    print("\n" + "=" * 70)
    print("DTC SCRAPING SUMMARY")
    print("=" * 70)

    print("\nPer-Source Results:")
    print("-" * 70)

    total_scraped = 0
    for scraper_name, codes, error in all_results:
        status = f"{len(codes)} codes" if not error else f"ERROR: {error}"
        print(f"  {scraper_name:20} : {status}")
        total_scraped += len(codes)

    print("-" * 70)
    print(f"  {'Total scraped':20} : {total_scraped} codes")

    print("\nMaster Database:")
    print("-" * 70)
    metadata = master_data.get("metadata", {})
    print(f"  Total codes        : {metadata.get('total_codes', 0)}")
    print(f"  Translated (HU)    : {metadata.get('translated', 0)}")
    print(f"  New codes added    : {metadata.get('new_codes_added', 0)}")
    print(f"  Codes updated      : {metadata.get('codes_updated', 0)}")

    # Category breakdown
    categories = {}
    for code in master_data.get("codes", []):
        cat = code.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    print("\nBy Category:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat:20} : {count}")

    print("=" * 70)
    print(f"Master file: {MASTER_FILE}")
    print("=" * 70 + "\n")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run all DTC scrapers and merge results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available sources:
  obd-codes     - obd-codes.com
  dtcbase       - dtcbase.com
  troublecodes  - troublecodes.net
  autocodes     - autocodes.com
  bba-reman     - bba-reman.com
  klavkarr      - klavkarr.com

Examples:
  python scripts/scrape_all_dtc_sources.py
  python scripts/scrape_all_dtc_sources.py --limit 100
  python scripts/scrape_all_dtc_sources.py --sources obd-codes,dtcbase
        """,
    )
    parser.add_argument(
        "--sources",
        type=str,
        help="Comma-separated list of sources to scrape (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of codes per source",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run scrapers sequentially instead of in parallel",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine which scrapers to run
    if args.sources:
        source_list = [s.strip() for s in args.sources.split(",")]
        scrapers_to_run = {k: v for k, v in SCRAPERS.items() if k in source_list}
        invalid = set(source_list) - set(SCRAPERS.keys())
        if invalid:
            logger.warning(f"Unknown sources: {invalid}")
    else:
        scrapers_to_run = SCRAPERS

    if not scrapers_to_run:
        logger.error("No valid scrapers to run")
        return

    logger.info(f"Running {len(scrapers_to_run)} scrapers: {', '.join(scrapers_to_run.keys())}")

    # Run scrapers
    all_results = []

    if args.sequential:
        # Run sequentially
        for name, module in scrapers_to_run.items():
            result = await run_scraper(name, module, limit=args.limit)
            all_results.append(result)
    else:
        # Run in parallel (be careful with rate limits)
        tasks = [
            run_scraper(name, module, limit=args.limit)
            for name, module in scrapers_to_run.items()
        ]
        all_results = await asyncio.gather(*tasks)

    # Merge results
    logger.info("Merging all results...")
    master_data = merge_all_codes(all_results)

    # Save master file
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(MASTER_FILE, "w", encoding="utf-8") as f:
        json.dump(master_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(master_data['codes'])} codes to {MASTER_FILE}")

    # Print summary
    print_summary(all_results, master_data)


if __name__ == "__main__":
    asyncio.run(main())
