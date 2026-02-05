#!/usr/bin/env python3
"""
OBD Trouble Codes Importer from mytrile/obd-trouble-codes GitHub Repository.

This script downloads and imports 11,000+ DTC codes from the comprehensive
mytrile/obd-trouble-codes repository.

Sources:
    - Primary: https://github.com/mytrile/obd-trouble-codes
    - Format: JSON/CSV
    - License: MIT (free to use)

Usage:
    python scripts/import_obd_codes.py --download     # Download and cache data
    python scripts/import_obd_codes.py --postgres     # Import to PostgreSQL only
    python scripts/import_obd_codes.py --neo4j        # Import to Neo4j only
    python scripts/import_obd_codes.py --all          # Download and import to all DBs
"""

import argparse
import asyncio
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Data paths
DATA_DIR = PROJECT_ROOT / "data" / "dtc_codes"
CACHE_FILE = DATA_DIR / "mytrile_codes.json"
MERGED_FILE = DATA_DIR / "all_codes.json"

# GitHub raw URLs for mytrile/obd-trouble-codes
# Repository: https://github.com/mytrile/obd-trouble-codes
# Files: obd-trouble-codes.json (395KB), obd-trouble-codes.csv, obd-trouble-codes.sqlite
MYTRILE_BASE_URL = "https://raw.githubusercontent.com/mytrile/obd-trouble-codes/master"

# Primary source - JSON file with all codes
MYTRILE_JSON_URL = f"{MYTRILE_BASE_URL}/obd-trouble-codes.json"

# Alternative: CSV file with all codes
MYTRILE_CSV_URL = f"{MYTRILE_BASE_URL}/obd-trouble-codes.csv"

# Legacy structure (no longer exists, kept for reference)
MYTRILE_FILES = {
    "powertrain": f"{MYTRILE_BASE_URL}/codes/powertrain.json",
    "body": f"{MYTRILE_BASE_URL}/codes/body.json",
    "chassis": f"{MYTRILE_BASE_URL}/codes/chassis.json",
    "network": f"{MYTRILE_BASE_URL}/codes/network.json",
}

# DTC Code patterns
DTC_PATTERN = re.compile(r'^[PCBU][0-9A-F]{4}$', re.IGNORECASE)


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


def get_system_from_code(code: str) -> str:
    """Determine the system from a DTC code."""
    if not code or len(code) < 3:
        return ""

    prefix = code[0].upper()
    middle = code[1:3]

    # Powertrain codes
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

    # Chassis codes
    elif prefix == "C":
        return "Chassis/ABS/Traction"

    # Body codes
    elif prefix == "B":
        return "Body/Interior"

    # Network codes
    elif prefix == "U":
        return "Network Communication"

    return ""


def get_severity_from_code(code: str) -> str:
    """Estimate severity based on DTC code pattern."""
    if not code:
        return "medium"

    prefix = code[0].upper()

    # Network and transmission codes are often more critical
    if prefix == "U":
        return "high"

    # Airbag/safety codes are critical
    if prefix == "B" and code.startswith(("B0", "B1")):
        return "critical"

    # Misfire and transmission codes are high priority
    if prefix == "P":
        if code.startswith("P03"):  # Misfire
            return "high"
        if code.startswith(("P07", "P08", "P09")):  # Transmission
            return "high"

    return "medium"


async def download_mytrile_codes() -> List[Dict[str, Any]]:
    """
    Download DTC codes from mytrile/obd-trouble-codes repository.

    The repo provides:
    - obd-trouble-codes.json (395KB) - Primary source
    - obd-trouble-codes.csv - Alternative format
    - obd-trouble-codes.sqlite - Database format

    Returns:
        List of DTC code dictionaries.
    """
    logger.info("Downloading codes from mytrile/obd-trouble-codes...")

    all_codes = []

    async with httpx.AsyncClient(timeout=60.0) as client:
        # Primary: Try JSON file first (most comprehensive)
        try:
            logger.info(f"Downloading JSON from {MYTRILE_JSON_URL}")
            response = await client.get(MYTRILE_JSON_URL)

            if response.status_code == 200:
                data = response.json()

                # The JSON structure is a dict where keys are DTC codes
                # e.g., {"P0001": "Fuel Volume Regulator Control Circuit/Open", ...}
                if isinstance(data, dict):
                    for code, description in data.items():
                        code_upper = code.upper().strip()
                        if DTC_PATTERN.match(code_upper):
                            all_codes.append({
                                "code": code_upper,
                                "description_en": description if isinstance(description, str) else str(description),
                                "source": "mytrile",
                            })

                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            code = item.get("code", "").upper().strip()
                            desc = item.get("description", item.get("description_en", ""))
                            if code and DTC_PATTERN.match(code):
                                all_codes.append({
                                    "code": code,
                                    "description_en": desc,
                                    "source": "mytrile",
                                })

                logger.info(f"Downloaded {len(all_codes)} codes from JSON")
            else:
                logger.warning(f"JSON download failed: HTTP {response.status_code}")

        except httpx.RequestError as e:
            logger.warning(f"JSON request error: {e}")
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error: {e}")

        # Fallback: Try CSV if JSON failed or empty
        if not all_codes:
            logger.info(f"Trying CSV format from {MYTRILE_CSV_URL}")
            try:
                response = await client.get(MYTRILE_CSV_URL)
                if response.status_code == 200:
                    all_codes = parse_csv_codes(response.text)
                    logger.info(f"Downloaded {len(all_codes)} codes from CSV")
            except Exception as e:
                logger.warning(f"CSV download failed: {e}")

    return all_codes


def parse_csv_codes(csv_content: str) -> List[Dict[str, Any]]:
    """
    Parse DTC codes from CSV content.

    Handles format: "P0100","Mass or Volume Air Flow Circuit Malfunction"

    Args:
        csv_content: Raw CSV content.

    Returns:
        List of DTC code dictionaries.
    """
    codes = []
    lines = csv_content.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Handle quoted CSV format: "P0100","Description text"
        # Remove leading/trailing quotes and split by ","
        if line.startswith('"'):
            # Parse properly quoted CSV
            parts = []
            current = ""
            in_quotes = False

            for char in line:
                if char == '"':
                    in_quotes = not in_quotes
                elif char == ',' and not in_quotes:
                    parts.append(current.strip())
                    current = ""
                else:
                    current += char

            parts.append(current.strip())

            if len(parts) >= 2:
                code = parts[0].upper().strip()
                description = parts[1].strip()
            else:
                continue
        else:
            # Simple comma-separated
            parts = line.split(",", 1)
            if len(parts) >= 2:
                code = parts[0].strip().upper().strip('"')
                description = parts[1].strip().strip('"')
            else:
                continue

        # Validate and add
        if code and description and DTC_PATTERN.match(code):
            codes.append({
                "code": code,
                "description_en": description,
                "category": get_category_from_code(code),
                "source": "mytrile",
            })

    logger.info(f"Parsed {len(codes)} codes from CSV")
    return codes


def normalize_codes(codes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize DTC codes to a consistent format.

    Args:
        codes: List of raw DTC code dictionaries.

    Returns:
        List of normalized DTC code dictionaries.
    """
    normalized = []
    seen_codes = set()

    for code_data in codes:
        # Extract code
        code = code_data.get("code", "").upper().strip()

        # Skip if invalid or duplicate
        if not DTC_PATTERN.match(code) or code in seen_codes:
            continue

        seen_codes.add(code)

        # Extract description
        description_en = (
            code_data.get("description_en") or
            code_data.get("description") or
            code_data.get("desc") or
            code_data.get("name") or
            ""
        ).strip()

        if not description_en:
            continue

        # Build normalized entry
        normalized.append({
            "code": code,
            "description_en": description_en,
            "description_hu": None,  # Will be filled by translation script
            "category": get_category_from_code(code),
            "severity": get_severity_from_code(code),
            "system": get_system_from_code(code),
            "is_generic": code[1] == "0",  # Generic codes have 0 as second character
            "symptoms": [],
            "possible_causes": [],
            "diagnostic_steps": [],
            "related_codes": [],
            "source": code_data.get("source", "mytrile"),
            "manufacturer": None,
            "translation_status": "pending",
        })

    # Sort by code
    normalized.sort(key=lambda x: x["code"])

    logger.info(f"Normalized {len(normalized)} unique codes")
    return normalized


def save_to_cache(codes: List[Dict[str, Any]], file_path: Path) -> None:
    """Save codes to a JSON cache file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "metadata": {
            "source": "mytrile/obd-trouble-codes",
            "downloaded_at": datetime.utcnow().isoformat(),
            "count": len(codes),
        },
        "codes": codes,
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(codes)} codes to {file_path}")


def load_from_cache(file_path: Path) -> List[Dict[str, Any]]:
    """Load codes from a JSON cache file."""
    if not file_path.exists():
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    codes = data.get("codes", [])
    logger.info(f"Loaded {len(codes)} codes from cache: {file_path}")
    return codes


def merge_with_existing(new_codes: List[Dict[str, Any]], existing_file: Path) -> List[Dict[str, Any]]:
    """
    Merge new codes with existing codes, preserving translations and custom data.

    Args:
        new_codes: List of newly downloaded codes.
        existing_file: Path to existing codes JSON file.

    Returns:
        Merged list of codes.
    """
    existing_codes = load_from_cache(existing_file) if existing_file.exists() else []

    # Create lookup dict for existing codes
    existing_lookup = {c["code"]: c for c in existing_codes}

    merged = []
    for new_code in new_codes:
        code = new_code["code"]

        if code in existing_lookup:
            # Preserve existing data, update with new info where missing
            existing = existing_lookup[code]
            merged_code = {**new_code}

            # Preserve translations and custom data
            if existing.get("description_hu"):
                merged_code["description_hu"] = existing["description_hu"]
                merged_code["translation_status"] = "completed"

            # Preserve symptoms, causes, steps if present
            for field in ["symptoms", "possible_causes", "diagnostic_steps", "related_codes"]:
                if existing.get(field):
                    merged_code[field] = existing[field]

            merged.append(merged_code)
        else:
            merged.append(new_code)

    # Add any existing codes not in new data
    new_code_set = {c["code"] for c in new_codes}
    for existing in existing_codes:
        if existing["code"] not in new_code_set:
            merged.append(existing)

    merged.sort(key=lambda x: x["code"])
    logger.info(f"Merged to {len(merged)} total codes")
    return merged


def get_sync_db_url() -> str:
    """Convert async database URL to sync for seeding."""
    from backend.app.core.config import settings
    url = settings.DATABASE_URL
    if url.startswith("postgresql+asyncpg://"):
        url = url.replace("postgresql+asyncpg://", "postgresql://")
    return url


def import_to_postgres(codes: List[Dict[str, Any]], batch_size: int = 100) -> int:
    """
    Import DTC codes to PostgreSQL database.

    Args:
        codes: List of DTC code dictionaries.
        batch_size: Number of codes per batch.

    Returns:
        Number of imported codes.
    """
    from backend.app.db.postgres.models import Base, DTCCode

    logger.info("Starting PostgreSQL import...")

    db_url = get_sync_db_url()
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)

    imported = 0
    skipped = 0

    with Session(engine) as session:
        for i in tqdm(range(0, len(codes), batch_size), desc="Importing to PostgreSQL"):
            batch = codes[i:i + batch_size]

            for code_data in batch:
                # Check if exists
                existing = session.query(DTCCode).filter_by(code=code_data["code"]).first()

                if existing:
                    # Update if we have more info
                    if code_data.get("description_hu") and not existing.description_hu:
                        existing.description_hu = code_data["description_hu"]
                    skipped += 1
                else:
                    # Insert new
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

    logger.info(f"PostgreSQL import complete: {imported} new, {skipped} existing")
    return imported


def import_to_neo4j(codes: List[Dict[str, Any]], batch_size: int = 50) -> int:
    """
    Import DTC codes to Neo4j database.

    Args:
        codes: List of DTC code dictionaries.
        batch_size: Number of codes per batch.

    Returns:
        Number of imported codes.
    """
    from backend.app.db.neo4j_models import DTCNode

    logger.info("Starting Neo4j import...")

    imported = 0
    skipped = 0

    for code_data in tqdm(codes, desc="Importing to Neo4j"):
        # Check if exists
        existing = DTCNode.nodes.get_or_none(code=code_data["code"])

        if existing:
            # Update if we have more info
            if code_data.get("description_hu") and not existing.description_hu:
                existing.description_hu = code_data["description_hu"]
                existing.save()
            skipped += 1
        else:
            # Create new node
            DTCNode(
                code=code_data["code"],
                description_en=code_data["description_en"],
                description_hu=code_data.get("description_hu"),
                category=code_data.get("category", "unknown"),
                severity=code_data.get("severity", "medium"),
                is_generic=str(code_data.get("is_generic", True)).lower(),
                system=code_data.get("system", ""),
            ).save()
            imported += 1

    logger.info(f"Neo4j import complete: {imported} new, {skipped} existing")
    return imported


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Import OBD-II trouble codes from mytrile/obd-trouble-codes repository"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download codes and save to cache",
    )
    parser.add_argument(
        "--postgres",
        action="store_true",
        help="Import to PostgreSQL database",
    )
    parser.add_argument(
        "--neo4j",
        action="store_true",
        help="Import to Neo4j database",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download and import to all databases",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use cached data instead of downloading",
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
    if not (args.download or args.postgres or args.neo4j or args.all):
        args.all = True

    try:
        # Get codes
        if args.use_cache and CACHE_FILE.exists():
            codes = load_from_cache(CACHE_FILE)
        else:
            # Download fresh data
            raw_codes = await download_mytrile_codes()

            if not raw_codes:
                logger.error("No codes downloaded. Check network connection.")
                sys.exit(1)

            # Normalize
            codes = normalize_codes(raw_codes)

            # Merge with existing (preserve translations)
            codes = merge_with_existing(codes, DATA_DIR / "generic_codes.json")

            # Save to cache
            save_to_cache(codes, CACHE_FILE)

        if not codes:
            logger.error("No codes available. Exiting.")
            sys.exit(1)

        logger.info(f"Total codes available: {len(codes)}")

        # Import to databases
        if args.postgres or args.all:
            import_to_postgres(codes)

        if args.neo4j or args.all:
            import_to_neo4j(codes)

        # Print summary
        print("\n" + "=" * 60)
        print("IMPORT SUMMARY")
        print("=" * 60)
        print(f"Total codes: {len(codes)}")

        # Category breakdown
        categories = {}
        for code in codes:
            cat = code.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

        print("\nBy category:")
        for cat, count in sorted(categories.items()):
            print(f"  {cat}: {count}")

        # Translation status
        translated = sum(1 for c in codes if c.get("description_hu"))
        print(f"\nTranslation status:")
        print(f"  Translated: {translated}")
        print(f"  Pending: {len(codes) - translated}")
        print("=" * 60)

        logger.info("Import completed successfully!")

    except Exception as e:
        logger.error(f"Import failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
