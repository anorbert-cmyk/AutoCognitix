#!/usr/bin/env python3
"""
Klavkarr DTC Database Scraper.

This script scrapes DTC codes from the Klavkarr database (www.klavkarr.com)
which contains 11,000+ categorized OBD-II trouble codes.

URL Pattern: https://www.klavkarr.com/data-trouble-code-obd2.php?dtc={range}

Ranges:
    - p0000-p0299, p0300-p0399, p0400-p0499, p0500-p0599
    - p0600-p0699, p0700-p0999, p1000-p1999
    - c0000-c0999, b0000-b0999, u0000-u0999

Usage:
    python scripts/scrape_klavkarr.py --scrape        # Scrape all ranges
    python scripts/scrape_klavkarr.py --postgres      # Import to PostgreSQL
    python scripts/scrape_klavkarr.py --all           # Scrape and import
"""

import argparse
import asyncio
import json
import logging
import html
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from bs4 import BeautifulSoup
from tqdm import tqdm

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
CACHE_FILE = DATA_DIR / "klavkarr_codes.json"

# Klavkarr URL configuration
BASE_URL = "https://www.klavkarr.com/data-trouble-code-obd2.php"

# DTC code ranges to scrape - Only P-series codes are publicly available on Klavkarr
# Note: C, B, U codes return "selection incorrect" errors - only available in paid software
DTC_RANGES = [
    # Powertrain generic (P0xxx) - Klavkarr's actual URL structure
    "p0000-p0299",   # Air/fuel mixture
    "p0300-p0399",   # Ignition/misfire
    "p0400-p0499",   # Auxiliary emissions
    "p0500-p0599",   # Speed/idle control
    "p0600-p0699",   # Computer output circuits
    "p0700-p0999",   # Transmission
    # Complete P-series (catch-all to ensure we get all codes)
    "pxxxx",         # All powertrain codes - includes ~1400+ codes
]

# Rate limiting configuration - 2 seconds between requests to be respectful
RATE_LIMIT_DELAY = 2.0  # seconds between requests
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


def get_category_from_code(code: str) -> str:
    """Determine the category from a DTC code prefix."""
    prefix = code[0].upper() if code else ""
    categories = {
        "P": "powertrain",
        "C": "chassis",
        "B": "body",
        "U": "network",
    }
    return categories.get(prefix, "unknown")


def get_severity_from_code(code: str) -> str:
    """Estimate severity based on DTC code pattern."""
    if not code:
        return "medium"

    prefix = code[0].upper()

    if prefix == "U":
        return "high"
    if prefix == "B" and code.startswith(("B0", "B1")):
        return "critical"
    if prefix == "P":
        if code.startswith("P03"):  # Misfire
            return "high"
        if code.startswith(("P07", "P08", "P09")):  # Transmission
            return "high"

    return "medium"


def get_system_from_code(code: str) -> str:
    """Determine the system from a DTC code."""
    if not code or len(code) < 3:
        return ""

    prefix = code[0].upper()
    middle = code[1:3]

    if prefix == "P":
        systems = {
            "00": "Fuel and Air Metering",
            "01": "Fuel and Air Metering",
            "02": "Fuel and Air Metering Injection",
            "03": "Ignition System/Misfire",
            "04": "Auxiliary Emission Controls",
            "05": "Vehicle Speed and Idle Control",
            "06": "Computer Output Circuits",
            "07": "Transmission",
            "08": "Transmission",
            "09": "Transmission",
            "0A": "Hybrid Propulsion",
        }
        return systems.get(middle, "Powertrain")
    elif prefix == "C":
        return "Chassis/ABS/Traction"
    elif prefix == "B":
        return "Body/Interior"
    elif prefix == "U":
        return "Network Communication"

    return ""


async def scrape_range(
    client: httpx.AsyncClient,
    dtc_range: str,
    retry_count: int = 0,
) -> List[Dict[str, Any]]:
    """
    Scrape DTC codes from a specific range page.

    Args:
        client: HTTP client instance.
        dtc_range: DTC range string (e.g., "p0000-p0299").
        retry_count: Current retry attempt.

    Returns:
        List of scraped DTC code dictionaries.
    """
    url = f"{BASE_URL}?dtc={dtc_range}"
    codes = []

    try:
        response = await client.get(url)

        if response.status_code == 403:
            logger.warning(f"Access forbidden for {dtc_range} - may be rate limited")
            if retry_count < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY * (retry_count + 1))
                return await scrape_range(client, dtc_range, retry_count + 1)
            return codes

        if response.status_code != 200:
            logger.warning(f"HTTP {response.status_code} for {dtc_range}")
            return codes

        soup = BeautifulSoup(response.text, "html.parser")

        # Find the table with DTC codes
        # Klavkarr typically uses tables with code and description columns
        tables = soup.find_all("table")

        for table in tables:
            rows = table.find_all("tr")

            for row in rows:
                cells = row.find_all(["td", "th"])

                if len(cells) >= 2:
                    code_cell = cells[0].get_text(strip=True).upper()
                    desc_cell = cells[1].get_text(strip=True)

                    # Validate DTC code format
                    if re.match(r'^[PCBU][0-9A-F]{4}$', code_cell, re.IGNORECASE):
                        codes.append({
                            "code": code_cell,
                            "description_en": desc_cell,
                            "source": "klavkarr",
                        })

        # Alternative: look for list items or divs with specific patterns
        if not codes:
            # Try finding codes in div/span elements
            for element in soup.find_all(["div", "span", "p"]):
                text = element.get_text(strip=True)
                match = re.match(r'^([PCBU][0-9A-F]{4})\s*[-:]\s*(.+)$', text, re.IGNORECASE)
                if match:
                    codes.append({
                        "code": match.group(1).upper(),
                        "description_en": match.group(2).strip(),
                        "source": "klavkarr",
                    })

        logger.debug(f"Scraped {len(codes)} codes from {dtc_range}")

    except httpx.RequestError as e:
        logger.warning(f"Request error for {dtc_range}: {e}")
        if retry_count < MAX_RETRIES:
            await asyncio.sleep(RETRY_DELAY * (retry_count + 1))
            return await scrape_range(client, dtc_range, retry_count + 1)
    except Exception as e:
        logger.error(f"Error scraping {dtc_range}: {e}")

    return codes


def log_progress(codes: List[Dict[str, Any]], dtc_range: str) -> None:
    """Log progress every 100 codes."""
    total = len(codes)
    if total > 0 and total % 100 == 0:
        logger.info(f"Progress: {total} codes scraped so far...")
    logger.info(f"Scraped {total} codes from range: {dtc_range}")


async def scrape_all_ranges() -> List[Dict[str, Any]]:
    """
    Scrape all DTC code ranges from Klavkarr.

    Returns:
        List of all scraped DTC code dictionaries.
    """
    logger.info(f"Starting Klavkarr scrape for {len(DTC_RANGES)} ranges...")
    logger.info(f"Rate limit: {RATE_LIMIT_DELAY} seconds between requests")

    all_codes = []

    # Use custom headers to appear as a browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    }

    async with httpx.AsyncClient(timeout=30.0, headers=headers, follow_redirects=True) as client:
        for dtc_range in tqdm(DTC_RANGES, desc="Scraping Klavkarr"):
            codes = await scrape_range(client, dtc_range)
            all_codes.extend(codes)

            # Rate limiting
            await asyncio.sleep(RATE_LIMIT_DELAY)

    logger.info(f"Scraped {len(all_codes)} codes total from Klavkarr")
    return all_codes


def sanitize_description(text: str, max_length: int = 1000) -> str:
    """
    Sanitize description text from external sources.

    Args:
        text: Input text to sanitize.
        max_length: Maximum allowed length.

    Returns:
        Sanitized text string.
    """
    if not text:
        return ""

    # Strip whitespace
    text = text.strip()

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Unescape HTML entities
    text = html.unescape(text)

    # Remove control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # Truncate to max length
    if len(text) > max_length:
        text = text[:max_length]

    return text


def normalize_codes(codes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize and deduplicate scraped codes.

    Args:
        codes: List of raw scraped codes.

    Returns:
        List of normalized, deduplicated codes.
    """
    seen = set()
    normalized = []

    for code_data in codes:
        code = code_data.get("code", "").upper().strip()

        if not code or code in seen:
            continue

        if not re.match(r'^[PCBU][0-9A-F]{4}$', code, re.IGNORECASE):
            continue

        seen.add(code)

        # Sanitize description to prevent injection attacks
        description = sanitize_description(code_data.get("description_en", ""))
        if not description:
            continue

        normalized.append({
            "code": code,
            "description_en": description,
            "description_hu": None,
            "category": get_category_from_code(code),
            "severity": get_severity_from_code(code),
            "system": get_system_from_code(code),
            "is_generic": code[1] == "0",  # Boolean, not string!
            "symptoms": [],
            "possible_causes": [],
            "diagnostic_steps": [],
            "related_codes": [],
            "source": "klavkarr",
            "manufacturer": None,
            "translation_status": "pending",
        })

    normalized.sort(key=lambda x: x["code"])
    logger.info(f"Normalized to {len(normalized)} unique codes")
    return normalized


def save_to_cache(codes: List[Dict[str, Any]], file_path: Path) -> None:
    """Save codes to a JSON cache file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "metadata": {
            "source": "klavkarr.com",
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "count": len(codes),
            "ranges_scraped": len(DTC_RANGES),
        },
        "codes": codes,
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(codes)} codes to {file_path}")


def load_from_cache(file_path: Path) -> List[Dict[str, Any]]:
    """Load codes from cache file."""
    if not file_path.exists():
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    codes = data.get("codes", [])
    logger.info(f"Loaded {len(codes)} codes from cache")
    return codes


def merge_with_master(
    new_codes: List[Dict[str, Any]],
    master_file: Path,
) -> Tuple[int, int]:
    """
    Merge scraped codes with master codes file.

    Args:
        new_codes: Newly scraped codes.
        master_file: Path to master codes JSON file.

    Returns:
        Tuple of (new_count, updated_count).
    """
    if not master_file.exists():
        logger.warning(f"Master file not found: {master_file}")
        return 0, 0

    with open(master_file, "r", encoding="utf-8") as f:
        master_data = json.load(f)

    master_codes = {c["code"]: c for c in master_data.get("codes", [])}

    new_count = 0
    updated_count = 0

    for code_data in new_codes:
        code = code_data["code"]

        if code in master_codes:
            # Check if Klavkarr has better description
            existing_desc = master_codes[code].get("description_en", "")
            new_desc = code_data.get("description_en", "")

            if len(new_desc) > len(existing_desc):
                master_codes[code]["description_en"] = new_desc
                updated_count += 1
        else:
            master_codes[code] = code_data
            new_count += 1

    # Save updated master
    master_data["codes"] = sorted(master_codes.values(), key=lambda x: x["code"])
    master_data["metadata"]["count"] = len(master_data["codes"])
    master_data["metadata"]["last_updated"] = datetime.utcnow().isoformat()

    with open(master_file, "w", encoding="utf-8") as f:
        json.dump(master_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Merged: {new_count} new, {updated_count} updated")
    return new_count, updated_count


def get_sync_db_url() -> str:
    """Convert async database URL to sync."""
    from backend.app.core.config import settings
    url = settings.DATABASE_URL
    if url.startswith("postgresql+asyncpg://"):
        url = url.replace("postgresql+asyncpg://", "postgresql://")
    return url


def import_to_postgres(codes: List[Dict[str, Any]], batch_size: int = 100) -> int:
    """Import codes to PostgreSQL."""
    from backend.app.db.postgres.models import Base, DTCCode
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    logger.info("Importing to PostgreSQL...")

    db_url = get_sync_db_url()
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)

    imported = 0

    with Session(engine) as session:
        for i in tqdm(range(0, len(codes), batch_size), desc="PostgreSQL import"):
            batch = codes[i:i + batch_size]

            for code_data in batch:
                existing = session.query(DTCCode).filter_by(code=code_data["code"]).first()

                if not existing:
                    dtc = DTCCode(
                        code=code_data["code"],
                        description_en=code_data["description_en"],
                        description_hu=code_data.get("description_hu"),
                        category=code_data.get("category", "unknown"),
                        severity=code_data.get("severity", "medium"),
                        is_generic=code_data.get("is_generic", True),
                        system=code_data.get("system", ""),
                        symptoms=code_data.get("symptoms", []),
                        possible_causes=code_data.get("possible_causes", []),
                        diagnostic_steps=code_data.get("diagnostic_steps", []),
                        related_codes=code_data.get("related_codes", []),
                    )
                    session.add(dtc)
                    imported += 1

            session.commit()

    logger.info(f"Imported {imported} new codes to PostgreSQL")
    return imported


def import_to_neo4j(codes: List[Dict[str, Any]]) -> int:
    """Import codes to Neo4j."""
    from backend.app.db.neo4j_models import DTCNode

    logger.info("Importing to Neo4j...")

    imported = 0

    for code_data in tqdm(codes, desc="Neo4j import"):
        existing = DTCNode.nodes.get_or_none(code=code_data["code"])

        if not existing:
            DTCNode(
                code=code_data["code"],
                description_en=code_data["description_en"],
                description_hu=code_data.get("description_hu"),
                category=code_data.get("category", "unknown"),
                severity=code_data.get("severity", "medium"),
                is_generic=bool(code_data.get("is_generic", True)),  # Keep as boolean!
                system=code_data.get("system", ""),
            ).save()
            imported += 1

    logger.info(f"Imported {imported} new codes to Neo4j")
    return imported


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Scrape DTC codes from Klavkarr database"
    )
    parser.add_argument(
        "--scrape",
        action="store_true",
        help="Scrape codes from Klavkarr website",
    )
    parser.add_argument(
        "--postgres",
        action="store_true",
        help="Import to PostgreSQL",
    )
    parser.add_argument(
        "--neo4j",
        action="store_true",
        help="Import to Neo4j",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Scrape and import to all databases",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use cached data instead of scraping",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge scraped codes with master file",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Default to --all if nothing specified
    if not (args.scrape or args.postgres or args.neo4j or args.all):
        args.all = True

    try:
        # Get codes
        if args.use_cache and CACHE_FILE.exists():
            codes = load_from_cache(CACHE_FILE)
        elif args.scrape or args.all:
            raw_codes = await scrape_all_ranges()
            codes = normalize_codes(raw_codes)
            save_to_cache(codes, CACHE_FILE)
        else:
            codes = load_from_cache(CACHE_FILE)

        if not codes:
            logger.warning("No codes available.")
            return

        logger.info(f"Total codes: {len(codes)}")

        # Merge with master if requested
        if args.merge:
            master_file = DATA_DIR / "mytrile_codes.json"
            if master_file.exists():
                merge_with_master(codes, master_file)

        # Import to databases
        if args.postgres or args.all:
            import_to_postgres(codes)

        if args.neo4j or args.all:
            import_to_neo4j(codes)

        # Print summary
        print("\n" + "=" * 60)
        print("KLAVKARR SCRAPE SUMMARY")
        print("=" * 60)
        print(f"Total codes scraped: {len(codes)}")

        categories = {}
        for code in codes:
            cat = code.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

        print("\nBy category:")
        for cat, count in sorted(categories.items()):
            print(f"  {cat}: {count}")
        print("=" * 60)

    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
