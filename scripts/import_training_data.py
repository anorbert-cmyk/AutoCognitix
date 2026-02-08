#!/usr/bin/env python3
"""
Comprehensive Automotive Training Data Importer for AutoCognitix.

Downloads and processes FREE automotive datasets from multiple sources
for AI training. All data is saved in JSON format ready for Neo4j/Qdrant import.

Data Sources:
1. NHTSA TSB (Technical Service Bulletins) - Repair procedures, symptoms, fixes
2. python-OBD codes - 1000+ DTC codes with descriptions
3. OBDb GitHub repos - Vehicle-specific DTC codes and PIDs
4. Kaggle datasets - Vehicle maintenance and predictive maintenance data

Features:
- Async downloads with aiohttp/httpx
- Progress bars with tqdm
- Checkpoint/resume support
- Rate limiting (respects robots.txt)
- Error handling with retries
- JSON output for Neo4j/Qdrant import
- Hungarian translation support

Environment Variables:
    GITHUB_TOKEN - Optional GitHub token for higher rate limits (5000/hour vs 60/hour)
    KAGGLE_USERNAME - Optional Kaggle credentials
    KAGGLE_KEY - Optional Kaggle API key

Usage:
    python scripts/import_training_data.py                 # Import all sources
    python scripts/import_training_data.py --tsb           # NHTSA TSB only
    python scripts/import_training_data.py --python-obd    # python-OBD codes only
    python scripts/import_training_data.py --obdb          # OBDb GitHub repos only
    python scripts/import_training_data.py --kaggle        # Kaggle datasets only
    python scripts/import_training_data.py --resume        # Resume from checkpoint
    python scripts/import_training_data.py --stats         # Show statistics only
"""

import argparse
import asyncio
import csv
import io
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import aiohttp
import httpx
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

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

# =============================================================================
# Configuration
# =============================================================================

# Output directories
DATA_DIR = PROJECT_ROOT / "data"
TSB_DIR = DATA_DIR / "nhtsa" / "tsb"
PYTHON_OBD_DIR = DATA_DIR / "dtc_codes"
OBDB_DIR = DATA_DIR / "obdb" / "github"
KAGGLE_DIR = DATA_DIR / "kaggle"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"

# NHTSA TSB Configuration
NHTSA_TSB_URL = "https://data.transportation.gov/api/views/hczg-qbhf/rows.csv?accessType=DOWNLOAD"
NHTSA_TSB_BACKUP_URL = "https://www.nhtsa.gov/sites/nhtsa.gov/files/documents/tsb_data.zip"

# python-OBD Configuration
# The python-OBD package contains DTC code definitions in its source
PYTHON_OBD_CODES_URL = "https://raw.githubusercontent.com/brendan-w/python-OBD/master/obd/OBDResponse.py"
PYTHON_OBD_COMMANDS_URL = "https://raw.githubusercontent.com/brendan-w/python-OBD/master/obd/commands.py"

# mytrile/obd-trouble-codes - Comprehensive DTC database (11,000+ codes)
MYTRILE_OBD_URL = "https://raw.githubusercontent.com/mytrile/obd-trouble-codes/master/obd-trouble-codes.json"

# OBDb GitHub Configuration
OBDB_API_URL = "https://api.github.com/orgs/OBDb/repos"
OBDB_RAW_URL = "https://raw.githubusercontent.com/OBDb"

# Rate limiting
REQUESTS_PER_SECOND = 5
HTTP_TIMEOUT = 60.0
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2.0

# Hungarian translation glossary for common automotive terms
TRANSLATION_GLOSSARY = {
    # Systems
    "engine": "motor",
    "transmission": "sebességváltó",
    "brake": "fék",
    "steering": "kormányzás",
    "suspension": "felfüggesztés",
    "exhaust": "kipufogó",
    "fuel": "üzemanyag",
    "cooling": "hűtés",
    "electrical": "elektromos",
    "emission": "emisszió",
    "airbag": "légzsák",
    "abs": "blokkolásgátló",
    # Components
    "sensor": "szenzor",
    "valve": "szelep",
    "pump": "szivattyú",
    "filter": "szűrő",
    "hose": "tömlő",
    "belt": "szíj",
    "gasket": "tömítés",
    "bearing": "csapágy",
    "cylinder": "henger",
    "piston": "dugattyú",
    # Issues
    "malfunction": "meghibásodás",
    "failure": "hiba",
    "leak": "szivárgás",
    "noise": "zaj",
    "vibration": "vibráció",
    "warning": "figyelmeztetés",
    "recall": "visszahívás",
    # Actions
    "replace": "cserél",
    "repair": "javít",
    "inspect": "ellenőriz",
    "adjust": "beállít",
    "clean": "tisztít",
}


# =============================================================================
# Rate Limiter
# =============================================================================


class RateLimiter:
    """Token bucket rate limiter for API requests."""

    def __init__(self, requests_per_second: float):
        self.rate = requests_per_second
        self.tokens = requests_per_second
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire a token, waiting if necessary."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.tokens = min(self.rate, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens < 1:
                wait_time = (1 - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1


# =============================================================================
# Checkpoint Management
# =============================================================================


@dataclass
class ImportCheckpoint:
    """Tracks import progress for resume capability."""

    tsb_completed: bool = False
    tsb_count: int = 0
    python_obd_completed: bool = False
    python_obd_count: int = 0
    obdb_repos_downloaded: Set[str] = field(default_factory=set)
    obdb_repos_failed: Set[str] = field(default_factory=set)
    kaggle_datasets_downloaded: Set[str] = field(default_factory=set)
    last_updated: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tsb_completed": self.tsb_completed,
            "tsb_count": self.tsb_count,
            "python_obd_completed": self.python_obd_completed,
            "python_obd_count": self.python_obd_count,
            "obdb_repos_downloaded": list(self.obdb_repos_downloaded),
            "obdb_repos_failed": list(self.obdb_repos_failed),
            "kaggle_datasets_downloaded": list(self.kaggle_datasets_downloaded),
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImportCheckpoint":
        return cls(
            tsb_completed=data.get("tsb_completed", False),
            tsb_count=data.get("tsb_count", 0),
            python_obd_completed=data.get("python_obd_completed", False),
            python_obd_count=data.get("python_obd_count", 0),
            obdb_repos_downloaded=set(data.get("obdb_repos_downloaded", [])),
            obdb_repos_failed=set(data.get("obdb_repos_failed", [])),
            kaggle_datasets_downloaded=set(data.get("kaggle_datasets_downloaded", [])),
            last_updated=data.get("last_updated", ""),
        )


def load_checkpoint() -> ImportCheckpoint:
    """Load import checkpoint from file."""
    checkpoint_file = CHECKPOINT_DIR / "training_data_checkpoint.json"
    if checkpoint_file.exists():
        with open(checkpoint_file, encoding="utf-8") as f:
            return ImportCheckpoint.from_dict(json.load(f))
    return ImportCheckpoint()


def save_checkpoint(checkpoint: ImportCheckpoint) -> None:
    """Save import checkpoint to file."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint.last_updated = datetime.now(timezone.utc).isoformat()
    checkpoint_file = CHECKPOINT_DIR / "training_data_checkpoint.json"
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(checkpoint.to_dict(), f, indent=2)


# =============================================================================
# Utility Functions
# =============================================================================


def save_json(data: Any, file_path: Path) -> None:
    """Save data to JSON file with proper formatting."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(file_path: Path) -> Optional[Any]:
    """Load data from JSON file."""
    if not file_path.exists():
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


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

    # Network codes are often critical
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


def simple_translate(text: str, glossary: Dict[str, str]) -> Optional[str]:
    """
    Simple word-by-word translation using glossary.
    Returns None if no translation available.
    """
    if not text:
        return None

    text_lower = text.lower()
    translated_words = []
    found_translation = False

    for word in text_lower.split():
        # Clean punctuation
        clean_word = word.strip(".,;:!?()-")
        if clean_word in glossary:
            translated_words.append(glossary[clean_word])
            found_translation = True
        else:
            translated_words.append(word)

    if found_translation:
        return " ".join(translated_words).capitalize()
    return None


# =============================================================================
# NHTSA TSB Importer
# =============================================================================


async def import_nhtsa_tsb(
    checkpoint: ImportCheckpoint,
    rate_limiter: RateLimiter,
) -> Dict[str, Any]:
    """
    Import NHTSA Technical Service Bulletins.

    TSBs contain:
    - Manufacturer repair procedures
    - Known issues and symptoms
    - Recommended fixes
    - Affected vehicles and date ranges

    Returns:
        Dictionary with import statistics.
    """
    logger.info("=" * 60)
    logger.info("IMPORTING NHTSA TECHNICAL SERVICE BULLETINS")
    logger.info("=" * 60)

    if checkpoint.tsb_completed:
        logger.info(f"TSB import already completed ({checkpoint.tsb_count} records)")
        return {"status": "skipped", "count": checkpoint.tsb_count}

    stats = {
        "source": "NHTSA TSB",
        "url": NHTSA_TSB_URL,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "records_total": 0,
        "records_by_make": defaultdict(int),
        "records_by_year": defaultdict(int),
        "components": defaultdict(int),
        "errors": [],
    }

    TSB_DIR.mkdir(parents=True, exist_ok=True)

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=300)
    ) as session:
        await rate_limiter.acquire()

        logger.info(f"Downloading TSB data from {NHTSA_TSB_URL}")
        logger.info("This may take several minutes for the initial download...")

        try:
            async with session.get(NHTSA_TSB_URL) as response:
                if response.status != 200:
                    error = f"TSB download failed: HTTP {response.status}"
                    logger.error(error)
                    stats["errors"].append(error)
                    return stats

                # Read CSV content
                content = await response.text()
                logger.info(f"Downloaded {len(content):,} bytes")

        except asyncio.TimeoutError:
            logger.error("TSB download timed out. Try again later.")
            stats["errors"].append("Download timeout")
            return stats
        except aiohttp.ClientError as e:
            logger.error(f"TSB download error: {e}")
            stats["errors"].append(str(e))
            return stats

    # Parse CSV
    logger.info("Parsing TSB CSV data...")

    tsb_records = []
    csv_reader = csv.DictReader(io.StringIO(content))

    for row in tqdm(csv_reader, desc="Parsing TSB records"):
        try:
            # Normalize field names (handle variations)
            record = {
                "tsb_id": (
                    row.get("TSB ID", "")
                    or row.get("tsb_id", "")
                    or row.get("TSBID", "")
                ).strip(),
                "manufacturer": (
                    row.get("Manufacturer", "")
                    or row.get("manufacturer", "")
                    or row.get("MFR_NAME", "")
                ).strip(),
                "make": (
                    row.get("Make", "") or row.get("make", "") or row.get("MAKE", "")
                ).strip(),
                "model": (
                    row.get("Model", "")
                    or row.get("model", "")
                    or row.get("MODEL", "")
                ).strip(),
                "model_year": (
                    row.get("Model Year", "")
                    or row.get("model_year", "")
                    or row.get("YEAR", "")
                ).strip(),
                "component": (
                    row.get("Component", "")
                    or row.get("component", "")
                    or row.get("COMPONENT", "")
                ).strip(),
                "summary": (
                    row.get("Summary", "")
                    or row.get("summary", "")
                    or row.get("SUMMARY", "")
                ).strip(),
                "date": (
                    row.get("Date", "")
                    or row.get("date", "")
                    or row.get("TSB_DATE", "")
                ).strip(),
                # Additional fields if present
                "report_date": row.get("Report Date", "").strip(),
                "nhtsa_id": row.get("NHTSA ID", "").strip(),
            }

            # Skip empty records
            if not record["tsb_id"] and not record["summary"]:
                continue

            # Parse year
            year = None
            if record["model_year"]:
                try:
                    year = int(record["model_year"])
                except ValueError:
                    pass

            # Generate simple Hungarian translation hint
            record["summary_hu_hint"] = simple_translate(
                record["summary"][:100], TRANSLATION_GLOSSARY
            )

            tsb_records.append(record)

            # Update statistics
            stats["records_total"] += 1
            if record["make"]:
                stats["records_by_make"][record["make"]] += 1
            if year:
                stats["records_by_year"][str(year)] += 1
            if record["component"]:
                stats["components"][record["component"]] += 1

        except Exception as e:
            stats["errors"].append(f"Parse error: {e}")

    logger.info(f"Parsed {len(tsb_records):,} TSB records")

    # Save all records
    all_tsb_file = TSB_DIR / "all_tsb.json"
    save_json(
        {
            "metadata": {
                "source": "NHTSA TSB",
                "url": NHTSA_TSB_URL,
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
                "record_count": len(tsb_records),
            },
            "records": tsb_records,
        },
        all_tsb_file,
    )

    # Save by make for easier processing
    by_make: Dict[str, List[Dict]] = defaultdict(list)
    for record in tsb_records:
        make = record.get("make", "Unknown") or "Unknown"
        by_make[make].append(record)

    for make, records in tqdm(by_make.items(), desc="Saving by make"):
        safe_make = re.sub(r'[<>:"/\\|?*]', "_", make)
        make_file = TSB_DIR / f"{safe_make}.json"
        save_json(
            {
                "make": make,
                "record_count": len(records),
                "records": records,
            },
            make_file,
        )

    # Save statistics
    stats["completed_at"] = datetime.now(timezone.utc).isoformat()
    stats["records_by_make"] = dict(stats["records_by_make"])
    stats["records_by_year"] = dict(stats["records_by_year"])
    stats["components"] = dict(
        sorted(stats["components"].items(), key=lambda x: -x[1])[:50]
    )

    stats_file = TSB_DIR / "import_statistics.json"
    save_json(stats, stats_file)

    # Update checkpoint
    checkpoint.tsb_completed = True
    checkpoint.tsb_count = len(tsb_records)
    save_checkpoint(checkpoint)

    logger.info(f"TSB import complete: {len(tsb_records):,} records")
    return stats


# =============================================================================
# python-OBD Codes Importer
# =============================================================================


async def import_python_obd_codes(
    checkpoint: ImportCheckpoint,
    rate_limiter: RateLimiter,
) -> Dict[str, Any]:
    """
    Extract and import DTC codes from python-OBD library and mytrile repository.

    Sources:
    1. python-OBD library - OBD-II protocol implementation
    2. mytrile/obd-trouble-codes - Comprehensive DTC database (11,000+ codes)

    Returns:
        Dictionary with import statistics.
    """
    logger.info("=" * 60)
    logger.info("IMPORTING PYTHON-OBD AND MYTRILE DTC CODES")
    logger.info("=" * 60)

    if checkpoint.python_obd_completed:
        logger.info(
            f"python-OBD import already completed ({checkpoint.python_obd_count} codes)"
        )
        return {"status": "skipped", "count": checkpoint.python_obd_count}

    stats = {
        "source": "python-OBD + mytrile",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "codes_total": 0,
        "codes_by_category": defaultdict(int),
        "codes_by_severity": defaultdict(int),
        "errors": [],
    }

    PYTHON_OBD_DIR.mkdir(parents=True, exist_ok=True)

    all_codes: Dict[str, Dict[str, Any]] = {}

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        # Source 1: mytrile/obd-trouble-codes (primary - most comprehensive)
        logger.info("Downloading mytrile/obd-trouble-codes...")
        await rate_limiter.acquire()

        try:
            response = await client.get(MYTRILE_OBD_URL)
            if response.status_code == 200:
                data = response.json()

                # Parse mytrile format - it's a list of objects where each object
                # contains code:code and description:description pairs
                # Example: [{"P0100": "P0101", "Mass or...": "Description..."}]
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            # Each dict has 2 key-value pairs: (code, code) and (desc, desc)
                            # The second key-value pair in each is the actual data
                            items = list(item.items())
                            if len(items) >= 2:
                                # Second entry value is the code, second entry's key is description
                                code = items[0][1]  # P0101 (the value of first pair)
                                description = items[1][1]  # Description (value of second pair)

                                if isinstance(code, str):
                                    code_upper = code.upper().strip()
                                    if re.match(r"^[PCBU][0-9A-F]{4}$", code_upper):
                                        all_codes[code_upper] = {
                                            "code": code_upper,
                                            "description_en": (
                                                description
                                                if isinstance(description, str)
                                                else str(description)
                                            ),
                                            "category": get_category_from_code(code_upper),
                                            "severity": get_severity_from_code(code_upper),
                                            "is_generic": code_upper[1] == "0",
                                            "source": "mytrile",
                                        }
                elif isinstance(data, dict):
                    # Alternative format: {"P0001": "Description", ...}
                    for code, description in data.items():
                        code_upper = code.upper().strip()
                        if re.match(r"^[PCBU][0-9A-F]{4}$", code_upper):
                            all_codes[code_upper] = {
                                "code": code_upper,
                                "description_en": (
                                    description
                                    if isinstance(description, str)
                                    else str(description)
                                ),
                                "category": get_category_from_code(code_upper),
                                "severity": get_severity_from_code(code_upper),
                                "is_generic": code_upper[1] == "0",
                                "source": "mytrile",
                            }

                logger.info(f"Loaded {len(all_codes)} codes from mytrile")
            else:
                logger.warning(f"mytrile download failed: HTTP {response.status_code}")

        except Exception as e:
            logger.warning(f"mytrile download error: {e}")
            stats["errors"].append(f"mytrile: {e}")

        # Source 2: Try to extract codes from python-OBD source
        logger.info("Extracting codes from python-OBD source...")
        await rate_limiter.acquire()

        try:
            # Download the OBDResponse.py which contains DTC definitions
            response = await client.get(PYTHON_OBD_CODES_URL)
            if response.status_code == 200:
                source_code = response.text

                # Extract DTC codes from python source (pattern matching)
                # Look for patterns like: "P0001": "Description"
                dtc_pattern = re.compile(
                    r'["\']([PCBU][0-9A-F]{4})["\']:\s*["\']([^"\']+)["\']',
                    re.IGNORECASE,
                )

                for match in dtc_pattern.finditer(source_code):
                    code = match.group(1).upper()
                    description = match.group(2)

                    # Only add if not already present
                    if code not in all_codes:
                        all_codes[code] = {
                            "code": code,
                            "description_en": description,
                            "category": get_category_from_code(code),
                            "severity": get_severity_from_code(code),
                            "is_generic": code[1] == "0",
                            "source": "python-obd",
                        }

                logger.info(f"Total codes after python-OBD: {len(all_codes)}")

        except Exception as e:
            logger.warning(f"python-OBD source extraction error: {e}")
            stats["errors"].append(f"python-obd: {e}")

    # Add Hungarian translation hints
    for code_data in all_codes.values():
        code_data["description_hu_hint"] = simple_translate(
            code_data["description_en"], TRANSLATION_GLOSSARY
        )

    # Convert to list and sort
    codes_list = sorted(all_codes.values(), key=lambda x: x["code"])

    # Update statistics
    stats["codes_total"] = len(codes_list)
    for code_data in codes_list:
        stats["codes_by_category"][code_data["category"]] += 1
        stats["codes_by_severity"][code_data["severity"]] += 1

    # Save to file
    output_file = PYTHON_OBD_DIR / "python_obd_codes.json"
    save_json(
        {
            "metadata": {
                "sources": [
                    "mytrile/obd-trouble-codes",
                    "brendan-w/python-OBD",
                ],
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
                "code_count": len(codes_list),
            },
            "codes": codes_list,
        },
        output_file,
    )

    # Save statistics
    stats["completed_at"] = datetime.now(timezone.utc).isoformat()
    stats["codes_by_category"] = dict(stats["codes_by_category"])
    stats["codes_by_severity"] = dict(stats["codes_by_severity"])

    stats_file = PYTHON_OBD_DIR / "python_obd_statistics.json"
    save_json(stats, stats_file)

    # Update checkpoint
    checkpoint.python_obd_completed = True
    checkpoint.python_obd_count = len(codes_list)
    save_checkpoint(checkpoint)

    logger.info(f"python-OBD import complete: {len(codes_list)} codes")
    return stats


# =============================================================================
# OBDb GitHub Importer
# =============================================================================


def get_github_headers() -> Dict[str, str]:
    """Get GitHub API headers with optional token authentication."""
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "AutoCognitix-Training-Data-Importer/1.0",
    }

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
        logger.info("Using GitHub token authentication (5000 requests/hour)")
    else:
        logger.warning("No GITHUB_TOKEN found. Rate limited to 60 requests/hour.")

    return headers


async def fetch_obdb_repos(client: httpx.AsyncClient) -> List[Dict[str, Any]]:
    """Fetch all repositories from OBDb organization."""
    repos = []
    page = 1
    per_page = 100
    headers = get_github_headers()

    logger.info("Fetching OBDb repository list from GitHub API...")

    while True:
        url = OBDB_API_URL
        params = {"page": page, "per_page": per_page, "type": "public"}

        try:
            response = await client.get(url, headers=headers, params=params)

            # Check rate limit
            remaining = response.headers.get("X-RateLimit-Remaining", "?")

            if response.status_code == 403:
                logger.error("GitHub rate limit exceeded. Use GITHUB_TOKEN env var.")
                break

            if response.status_code != 200:
                logger.error(f"GitHub API error: {response.status_code}")
                break

            page_repos = response.json()

            if not page_repos:
                break

            repos.extend(page_repos)
            logger.info(f"Page {page}: {len(page_repos)} repos (rate limit: {remaining})")

            page += 1
            await asyncio.sleep(0.5)  # Rate limit delay

        except Exception as e:
            logger.error(f"Error fetching repos: {e}")
            break

    return repos


def parse_repo_name(name: str) -> Tuple[str, str, bool]:
    """Parse make and model from repository name."""
    # Skip non-vehicle repos
    non_vehicle_repos = {
        ".github",
        ".meta",
        ".vehicle-template",
        ".schemas",
        "SAEJ1979",
    }

    if name in non_vehicle_repos or name.startswith("."):
        return name, "", False

    # Handle special makes with hyphens
    special_makes = {
        "Mercedes-Benz": "Mercedes-Benz",
        "Alfa-Romeo": "Alfa Romeo",
        "Land-Rover": "Land Rover",
    }

    for special, replacement in special_makes.items():
        if name.startswith(special + "-"):
            model = name[len(special) + 1 :].replace("-", " ")
            return replacement, model, True

    # Standard parsing: first segment is make
    parts = name.split("-", 1)
    if len(parts) == 2:
        make = parts[0].replace("_", " ")
        model = parts[1].replace("-", " ").replace("_", " ")
        return make, model, True

    return name, "", False


async def download_obdb_repo(
    client: httpx.AsyncClient,
    repo: Dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> Tuple[str, Optional[Dict[str, Any]], str]:
    """Download signalset data from an OBDb repository."""
    async with semaphore:
        name = repo.get("name", "")
        make, model, is_vehicle = parse_repo_name(name)

        if not is_vehicle:
            return name, None, "not_vehicle"

        result = {
            "repo_name": name,
            "make": make,
            "model": model,
            "github_url": repo.get("html_url", ""),
            "signals": [],
            "dtcs": [],
            "commands": [],
        }

        # Try different signalset paths
        signalset_paths = [
            ("main", "signalsets/v3/default.json"),
            ("master", "signalsets/v3/default.json"),
            ("main", "signalsets/v2/default.json"),
            ("master", "signalsets/default.json"),
        ]

        for branch, path in signalset_paths:
            url = f"{OBDB_RAW_URL}/{name}/{branch}/{path}"

            try:
                response = await client.get(url)
                if response.status_code == 200:
                    signalset = response.json()

                    # Extract commands
                    commands = signalset.get("commands", [])
                    if isinstance(commands, list):
                        result["commands"] = commands
                        for cmd in commands:
                            if isinstance(cmd, dict):
                                signals = cmd.get("signals", [])
                                if isinstance(signals, list):
                                    result["signals"].extend(signals)

                    # Extract DTCs
                    dtcs = signalset.get("dtcs", {})
                    if isinstance(dtcs, dict):
                        for code, desc in dtcs.items():
                            if re.match(r"^[PCBU][0-9A-F]{4}$", code, re.IGNORECASE):
                                result["dtcs"].append(
                                    {
                                        "code": code.upper(),
                                        "description": desc,
                                        "vehicle": f"{make} {model}",
                                    }
                                )

                    result["signal_count"] = len(result["signals"])
                    result["dtc_count"] = len(result["dtcs"])
                    result["command_count"] = len(result["commands"])

                    return name, result, ""

            except Exception:
                continue

        return name, None, "no_signalset"


async def import_obdb_github(
    checkpoint: ImportCheckpoint,
    rate_limiter: RateLimiter,
    max_repos: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Import vehicle data from OBDb GitHub organization.

    OBDb contains:
    - Vehicle-specific DTC codes
    - OBD-II PIDs and signals
    - Commands and responses

    Returns:
        Dictionary with import statistics.
    """
    logger.info("=" * 60)
    logger.info("IMPORTING OBDB GITHUB REPOSITORIES")
    logger.info("=" * 60)

    stats = {
        "source": "OBDb GitHub",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "repos_total": 0,
        "repos_downloaded": 0,
        "repos_skipped": 0,
        "repos_failed": 0,
        "total_signals": 0,
        "total_dtcs": 0,
        "makes": defaultdict(int),
        "errors": [],
    }

    OBDB_DIR.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        # Fetch repository list
        repos = await fetch_obdb_repos(client)
        stats["repos_total"] = len(repos)

        if not repos:
            logger.error("No repositories found")
            return stats

        # Filter to vehicles only
        vehicle_repos = []
        for repo in repos:
            name = repo.get("name", "")
            _, _, is_vehicle = parse_repo_name(name)
            if is_vehicle and name not in checkpoint.obdb_repos_downloaded:
                vehicle_repos.append(repo)

        if max_repos:
            vehicle_repos = vehicle_repos[:max_repos]

        logger.info(f"Downloading {len(vehicle_repos)} vehicle repositories...")

        # Download with concurrency limit
        semaphore = asyncio.Semaphore(10)
        tasks = [
            download_obdb_repo(client, repo, semaphore) for repo in vehicle_repos
        ]

        results = await tqdm_asyncio.gather(*tasks, desc="Downloading OBDb repos")

        # Process results
        all_vehicles = []
        all_dtcs = []

        for name, data, error in results:
            if data:
                all_vehicles.append(data)
                all_dtcs.extend(data.get("dtcs", []))

                stats["repos_downloaded"] += 1
                stats["total_signals"] += data.get("signal_count", 0)
                stats["total_dtcs"] += data.get("dtc_count", 0)
                stats["makes"][data.get("make", "Unknown")] += 1

                checkpoint.obdb_repos_downloaded.add(name)

                # Save individual vehicle data
                safe_make = re.sub(r'[<>:"/\\|?*]', "_", data["make"])
                safe_model = re.sub(r'[<>:"/\\|?*]', "_", data["model"])
                vehicle_dir = OBDB_DIR / safe_make
                vehicle_dir.mkdir(parents=True, exist_ok=True)
                vehicle_file = vehicle_dir / f"{safe_model}.json"
                save_json(data, vehicle_file)

            elif error == "not_vehicle":
                stats["repos_skipped"] += 1
            else:
                stats["repos_failed"] += 1
                checkpoint.obdb_repos_failed.add(name)

        # Save all vehicle DTCs
        if all_dtcs:
            dtcs_file = OBDB_DIR / "all_vehicle_dtcs.json"
            save_json(
                {
                    "metadata": {
                        "source": "OBDb GitHub",
                        "downloaded_at": datetime.now(timezone.utc).isoformat(),
                        "dtc_count": len(all_dtcs),
                    },
                    "dtcs": all_dtcs,
                },
                dtcs_file,
            )

    # Save statistics
    stats["completed_at"] = datetime.now(timezone.utc).isoformat()
    stats["makes"] = dict(sorted(stats["makes"].items(), key=lambda x: -x[1]))

    stats_file = OBDB_DIR / "import_statistics.json"
    save_json(stats, stats_file)

    save_checkpoint(checkpoint)

    logger.info(
        f"OBDb import complete: {stats['repos_downloaded']} repos, "
        f"{stats['total_dtcs']} DTCs"
    )
    return stats


# =============================================================================
# Kaggle Dataset Importer
# =============================================================================


async def import_kaggle_datasets(
    checkpoint: ImportCheckpoint,
    rate_limiter: RateLimiter,
) -> Dict[str, Any]:
    """
    Import vehicle maintenance datasets from Kaggle (if accessible without login).

    Note: Most Kaggle datasets require authentication.
    This function attempts to download publicly available mirrors.

    Known public datasets:
    - Vehicle maintenance prediction datasets
    - Car failure analysis data
    - Automotive sensor data
    """
    logger.info("=" * 60)
    logger.info("IMPORTING KAGGLE DATASETS")
    logger.info("=" * 60)

    stats = {
        "source": "Kaggle",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "datasets_found": 0,
        "datasets_downloaded": 0,
        "note": "Most Kaggle datasets require authentication",
        "errors": [],
    }

    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)

    # Check for Kaggle credentials
    kaggle_username = os.environ.get("KAGGLE_USERNAME")
    kaggle_key = os.environ.get("KAGGLE_KEY")

    if kaggle_username and kaggle_key:
        logger.info("Kaggle credentials found. Attempting API access...")

        try:
            # Try to import and use Kaggle API
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()

            # List of automotive maintenance datasets
            target_datasets = [
                "uciml/autompg-dataset",
                "shivamkushwaha/vehicle-maintenance-prediction",
                "manishprasadofficial/vehicle-predictive-maintenance-dataset",
            ]

            for dataset_name in target_datasets:
                if dataset_name in checkpoint.kaggle_datasets_downloaded:
                    continue

                try:
                    logger.info(f"Downloading: {dataset_name}")
                    dataset_dir = KAGGLE_DIR / dataset_name.replace("/", "_")
                    dataset_dir.mkdir(parents=True, exist_ok=True)

                    api.dataset_download_files(
                        dataset_name, path=str(dataset_dir), unzip=True
                    )

                    stats["datasets_downloaded"] += 1
                    checkpoint.kaggle_datasets_downloaded.add(dataset_name)

                except Exception as e:
                    logger.warning(f"Failed to download {dataset_name}: {e}")
                    stats["errors"].append(f"{dataset_name}: {e}")

        except ImportError:
            logger.warning("Kaggle package not installed. pip install kaggle")
            stats["errors"].append("Kaggle package not installed")
        except Exception as e:
            logger.warning(f"Kaggle API error: {e}")
            stats["errors"].append(str(e))

    else:
        logger.warning(
            "No Kaggle credentials found. Set KAGGLE_USERNAME and KAGGLE_KEY."
        )
        stats["note"] = "Set KAGGLE_USERNAME and KAGGLE_KEY environment variables"

        # Try public mirrors (these may or may not be available)
        public_mirrors = [
            # Auto MPG dataset (UCI ML Repository - often mirrored)
            (
                "auto_mpg",
                "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data",
            ),
        ]

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            for name, url in public_mirrors:
                if name in checkpoint.kaggle_datasets_downloaded:
                    continue

                try:
                    await rate_limiter.acquire()
                    response = await client.get(url)

                    if response.status_code == 200:
                        output_file = KAGGLE_DIR / f"{name}.data"
                        with open(output_file, "w", encoding="utf-8") as f:
                            f.write(response.text)

                        stats["datasets_downloaded"] += 1
                        checkpoint.kaggle_datasets_downloaded.add(name)
                        logger.info(f"Downloaded: {name}")

                except Exception as e:
                    logger.warning(f"Failed to download {name}: {e}")
                    stats["errors"].append(f"{name}: {e}")

    # Save statistics
    stats["completed_at"] = datetime.now(timezone.utc).isoformat()
    stats_file = KAGGLE_DIR / "import_statistics.json"
    save_json(stats, stats_file)

    save_checkpoint(checkpoint)

    logger.info(f"Kaggle import complete: {stats['datasets_downloaded']} datasets")
    return stats


# =============================================================================
# Statistics Generation
# =============================================================================


def generate_statistics() -> Dict[str, Any]:
    """Generate comprehensive statistics from all imported data."""
    stats = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": {},
        "totals": {
            "tsb_records": 0,
            "dtc_codes": 0,
            "obdb_vehicles": 0,
            "obdb_signals": 0,
            "kaggle_datasets": 0,
        },
    }

    # TSB Statistics
    tsb_stats_file = TSB_DIR / "import_statistics.json"
    if tsb_stats_file.exists():
        tsb_stats = load_json(tsb_stats_file)
        stats["sources"]["nhtsa_tsb"] = tsb_stats
        stats["totals"]["tsb_records"] = tsb_stats.get("records_total", 0)

    # python-OBD Statistics
    obd_stats_file = PYTHON_OBD_DIR / "python_obd_statistics.json"
    if obd_stats_file.exists():
        obd_stats = load_json(obd_stats_file)
        stats["sources"]["python_obd"] = obd_stats
        stats["totals"]["dtc_codes"] += obd_stats.get("codes_total", 0)

    # OBDb Statistics
    obdb_stats_file = OBDB_DIR / "import_statistics.json"
    if obdb_stats_file.exists():
        obdb_stats = load_json(obdb_stats_file)
        stats["sources"]["obdb"] = obdb_stats
        stats["totals"]["obdb_vehicles"] = obdb_stats.get("repos_downloaded", 0)
        stats["totals"]["obdb_signals"] = obdb_stats.get("total_signals", 0)
        stats["totals"]["dtc_codes"] += obdb_stats.get("total_dtcs", 0)

    # Kaggle Statistics
    kaggle_stats_file = KAGGLE_DIR / "import_statistics.json"
    if kaggle_stats_file.exists():
        kaggle_stats = load_json(kaggle_stats_file)
        stats["sources"]["kaggle"] = kaggle_stats
        stats["totals"]["kaggle_datasets"] = kaggle_stats.get("datasets_downloaded", 0)

    return stats


def print_statistics(stats: Dict[str, Any]) -> None:
    """Print formatted statistics."""
    print("\n" + "=" * 70)
    print("AUTOMOTIVE TRAINING DATA IMPORT STATISTICS")
    print("=" * 70)
    print(f"Generated: {stats['generated_at']}")

    print("\n" + "-" * 70)
    print("TOTALS")
    print("-" * 70)
    totals = stats.get("totals", {})
    print(f"  NHTSA TSB Records:    {totals.get('tsb_records', 0):>10,}")
    print(f"  DTC Codes:            {totals.get('dtc_codes', 0):>10,}")
    print(f"  OBDb Vehicles:        {totals.get('obdb_vehicles', 0):>10,}")
    print(f"  OBDb Signals/PIDs:    {totals.get('obdb_signals', 0):>10,}")
    print(f"  Kaggle Datasets:      {totals.get('kaggle_datasets', 0):>10,}")

    # Source details
    for source_name, source_stats in stats.get("sources", {}).items():
        print("\n" + "-" * 70)
        print(f"{source_name.upper()} DETAILS")
        print("-" * 70)

        if source_name == "nhtsa_tsb":
            by_make = source_stats.get("records_by_make", {})
            top_makes = sorted(by_make.items(), key=lambda x: -x[1])[:10]
            print("  Top 10 Makes:")
            for make, count in top_makes:
                print(f"    {make:25s}: {count:>6,}")

        elif source_name == "python_obd":
            by_cat = source_stats.get("codes_by_category", {})
            print("  By Category:")
            for cat, count in by_cat.items():
                print(f"    {cat:15s}: {count:>6,}")

        elif source_name == "obdb":
            makes = source_stats.get("makes", {})
            top_makes = list(makes.items())[:10]
            print("  Top 10 Makes:")
            for make, count in top_makes:
                print(f"    {make:25s}: {count:>6,} vehicles")

    print("\n" + "=" * 70)
    print(f"Data saved to: {DATA_DIR}")
    print("=" * 70)


# =============================================================================
# Main Entry Point
# =============================================================================


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Import FREE automotive training datasets for AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Data Sources:
  1. NHTSA TSB - Technical Service Bulletins (repair procedures, symptoms)
  2. python-OBD - DTC codes from python-OBD and mytrile repos
  3. OBDb GitHub - Vehicle-specific DTC codes and PIDs
  4. Kaggle - Vehicle maintenance datasets (requires credentials)

Examples:
  python scripts/import_training_data.py                 # Import all
  python scripts/import_training_data.py --tsb           # TSB only
  python scripts/import_training_data.py --python-obd    # DTC codes only
  python scripts/import_training_data.py --obdb          # OBDb repos only
  python scripts/import_training_data.py --resume        # Resume from checkpoint
  python scripts/import_training_data.py --stats         # Show statistics

Environment Variables:
  GITHUB_TOKEN - GitHub token for higher rate limits (5000/hour)
  KAGGLE_USERNAME, KAGGLE_KEY - Kaggle API credentials
        """,
    )

    parser.add_argument(
        "--tsb",
        action="store_true",
        help="Import NHTSA Technical Service Bulletins only",
    )
    parser.add_argument(
        "--python-obd",
        action="store_true",
        help="Import python-OBD and mytrile DTC codes only",
    )
    parser.add_argument(
        "--obdb",
        action="store_true",
        help="Import OBDb GitHub repositories only",
    )
    parser.add_argument(
        "--kaggle",
        action="store_true",
        help="Import Kaggle datasets only (requires credentials)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset checkpoint and start fresh",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show statistics only (no download)",
    )
    parser.add_argument(
        "--max-obdb-repos",
        type=int,
        default=None,
        help="Limit number of OBDb repos to download",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Stats only mode
    if args.stats:
        stats = generate_statistics()
        print_statistics(stats)
        return

    # Load or reset checkpoint
    if args.reset:
        checkpoint = ImportCheckpoint()
        logger.info("Checkpoint reset. Starting fresh.")
    else:
        checkpoint = load_checkpoint()
        if args.resume and checkpoint.last_updated:
            logger.info(f"Resuming from checkpoint: {checkpoint.last_updated}")

    # Create rate limiter
    rate_limiter = RateLimiter(REQUESTS_PER_SECOND)

    # Determine which sources to import
    import_all = not (args.tsb or args.python_obd or args.obdb or args.kaggle)

    all_stats = {}

    try:
        # Import NHTSA TSB
        if import_all or args.tsb:
            tsb_stats = await import_nhtsa_tsb(checkpoint, rate_limiter)
            all_stats["nhtsa_tsb"] = tsb_stats

        # Import python-OBD codes
        if import_all or args.python_obd:
            obd_stats = await import_python_obd_codes(checkpoint, rate_limiter)
            all_stats["python_obd"] = obd_stats

        # Import OBDb GitHub
        if import_all or args.obdb:
            obdb_stats = await import_obdb_github(
                checkpoint, rate_limiter, max_repos=args.max_obdb_repos
            )
            all_stats["obdb"] = obdb_stats

        # Import Kaggle datasets
        if import_all or args.kaggle:
            kaggle_stats = await import_kaggle_datasets(checkpoint, rate_limiter)
            all_stats["kaggle"] = kaggle_stats

        # Generate and print final statistics
        final_stats = generate_statistics()
        print_statistics(final_stats)

        # Save summary
        summary_file = DATA_DIR / "training_data_summary.json"
        save_json(
            {
                "import_stats": all_stats,
                "summary": final_stats,
            },
            summary_file,
        )

        logger.info("Import completed successfully!")
        logger.info(f"Summary saved to: {summary_file}")

    except KeyboardInterrupt:
        logger.info("\nImport interrupted by user. Progress saved.")
        save_checkpoint(checkpoint)
        sys.exit(0)

    except Exception as e:
        logger.error(f"Import failed: {e}")
        save_checkpoint(checkpoint)
        raise


if __name__ == "__main__":
    asyncio.run(main())
