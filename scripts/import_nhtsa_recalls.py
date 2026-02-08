#!/usr/bin/env python3
"""
NHTSA Recalls Import Script for AutoCognitix.

Downloads ALL vehicle recalls from the NHTSA (National Highway Traffic Safety
Administration) API from 1966 to 2024 for specified makes.

Features:
- Async HTTP client with aiohttp
- Rate limiting (configurable requests per second)
- Progress bars with tqdm
- Automatic retry with exponential backoff
- Saves data per year to data/nhtsa/recalls/
- Generates comprehensive statistics

API Endpoints Used:
- GET /recalls/recallsByVehicle?make={make}&model={model}&modelYear={year}
- GET /vehicles/GetModelsForMakeYear/make/{make}/modelyear/{year}

Usage:
    python scripts/import_nhtsa_recalls.py                  # Import all recalls
    python scripts/import_nhtsa_recalls.py --years 2020-2024  # Specific years
    python scripts/import_nhtsa_recalls.py --makes Toyota Honda  # Specific makes
    python scripts/import_nhtsa_recalls.py --stats-only       # Show stats only
    python scripts/import_nhtsa_recalls.py --verbose          # Verbose logging
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import aiohttp
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm

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

# NHTSA API endpoints
NHTSA_BASE_URL = "https://api.nhtsa.gov"
VPIC_BASE_URL = "https://vpic.nhtsa.dot.gov/api/vehicles"

# API Endpoints
RECALLS_ENDPOINT = f"{NHTSA_BASE_URL}/recalls/recallsByVehicle"
MODELS_ENDPOINT_TEMPLATE = f"{VPIC_BASE_URL}/GetModelsForMakeYear/make/{{make}}/modelyear/{{year}}"

# Output paths
DATA_DIR = PROJECT_ROOT / "data" / "nhtsa" / "recalls"
STATS_FILE = DATA_DIR / "import_statistics.json"
PROGRESS_FILE = DATA_DIR / "import_progress.json"

# Rate limiting - NHTSA allows ~10 requests/second
REQUESTS_PER_SECOND = 5
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2  # Exponential backoff base

# Year range
DEFAULT_START_YEAR = 1966
DEFAULT_END_YEAR = 2024

# Target makes
TARGET_MAKES = [
    "Toyota",
    "Honda",
    "Ford",
    "Chevrolet",
    "Nissan",
    "BMW",
    "Mercedes-Benz",
    "Volkswagen",
    "Audi",
    "Hyundai",
    "Kia",
    "Mazda",
    "Subaru",
    "Opel",  # Note: Opel may have limited US data
    "Skoda",  # Note: Skoda may have limited US data
]

# Max concurrent requests
MAX_CONCURRENT_REQUESTS = 10


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
# NHTSA API Client
# =============================================================================


class NHTSARecallsClient:
    """
    Async NHTSA API client for fetching recalls.

    Features:
    - Connection pooling with aiohttp
    - Rate limiting
    - Automatic retries with exponential backoff
    - Error handling
    """

    def __init__(
        self,
        requests_per_second: float = REQUESTS_PER_SECOND,
        timeout: float = REQUEST_TIMEOUT,
        max_retries: int = MAX_RETRIES,
    ):
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.rate_limiter = RateLimiter(requests_per_second)
        self.max_retries = max_retries
        self._session: Optional[aiohttp.ClientSession] = None
        self._request_count = 0
        self._error_count = 0

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                limit=MAX_CONCURRENT_REQUESTS,
                limit_per_host=MAX_CONCURRENT_REQUESTS,
            )
            self._session = aiohttp.ClientSession(
                timeout=self.timeout,
                connector=connector,
                headers={
                    "User-Agent": "AutoCognitix/1.0 (Vehicle Diagnostic Platform)",
                    "Accept": "application/json",
                },
            )
        return self._session

    async def _make_request(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Make an API request with rate limiting and retries.

        Args:
            url: Request URL.
            params: Query parameters.

        Returns:
            JSON response or None on error.
        """
        session = await self._get_session()

        for attempt in range(self.max_retries):
            await self.rate_limiter.acquire()
            self._request_count += 1

            try:
                async with session.get(url, params=params) as response:
                    if response.status == 429:
                        # Rate limited - wait and retry
                        retry_after = int(response.headers.get("Retry-After", 60))
                        logger.warning(
                            f"Rate limited, waiting {retry_after}s (attempt {attempt + 1})"
                        )
                        await asyncio.sleep(retry_after)
                        continue

                    if response.status == 404:
                        # No data found - this is normal
                        return {"results": [], "count": 0}

                    response.raise_for_status()
                    return await response.json()

            except aiohttp.ClientResponseError as e:
                logger.debug(f"HTTP error {e.status}: {url}")
                self._error_count += 1
                if attempt < self.max_retries - 1:
                    delay = RETRY_DELAY_BASE ** (attempt + 1)
                    await asyncio.sleep(delay)
                else:
                    return None

            except asyncio.TimeoutError:
                logger.debug(f"Timeout: {url}")
                self._error_count += 1
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(RETRY_DELAY_BASE)
                else:
                    return None

            except aiohttp.ClientError as e:
                logger.debug(f"Client error: {e}")
                self._error_count += 1
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(RETRY_DELAY_BASE)
                else:
                    return None

        return None

    async def get_models_for_make_year(
        self,
        make: str,
        year: int,
    ) -> List[str]:
        """
        Get available models for a make and year.

        Args:
            make: Vehicle make.
            year: Model year.

        Returns:
            List of model names.
        """
        url = MODELS_ENDPOINT_TEMPLATE.format(make=make, year=year)
        params = {"format": "json"}

        data = await self._make_request(url, params)
        if not data:
            return []

        results = data.get("Results", [])
        return [r.get("Model_Name", "") for r in results if r.get("Model_Name")]

    async def get_recalls_for_vehicle(
        self,
        make: str,
        model: str,
        year: int,
    ) -> List[Dict[str, Any]]:
        """
        Fetch recalls for a specific vehicle.

        Args:
            make: Vehicle make.
            model: Vehicle model.
            year: Model year.

        Returns:
            List of recall records.
        """
        params = {
            "make": make,
            "model": model,
            "modelYear": year,
        }

        data = await self._make_request(RECALLS_ENDPOINT, params)
        if not data:
            return []

        results = data.get("results", [])

        # Transform to standard format
        recalls = []
        for item in results:
            recall = {
                "campaign_number": item.get("NHTSACampaignNumber", ""),
                "manufacturer": item.get("Manufacturer", make),
                "make": make,
                "model": model,
                "model_year": year,
                "recall_date": item.get("ReportReceivedDate"),
                "component": item.get("Component", "Unknown"),
                "summary": item.get("Summary", ""),
                "consequence": item.get("Consequence", ""),
                "remedy": item.get("Remedy", ""),
                "notes": item.get("Notes", ""),
                "nhtsa_action_number": item.get("NHTSAActionNumber"),
                "potentially_affected_units": item.get("PotentialNumberofUnitsAffected"),
            }
            recalls.append(recall)

        return recalls

    def get_stats(self) -> Dict[str, int]:
        """Get request statistics."""
        return {
            "total_requests": self._request_count,
            "errors": self._error_count,
            "success_rate": (
                round((1 - self._error_count / max(1, self._request_count)) * 100, 2)
            ),
        }

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None


# =============================================================================
# Data Processing
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


def load_progress() -> Dict[str, Any]:
    """Load import progress state."""
    if PROGRESS_FILE.exists():
        return load_json(PROGRESS_FILE) or {}
    return {}


def save_progress(progress: Dict[str, Any]) -> None:
    """Save import progress state."""
    save_json(progress, PROGRESS_FILE)


# =============================================================================
# Import Logic
# =============================================================================


async def import_recalls_for_year(
    client: NHTSARecallsClient,
    make: str,
    year: int,
    pbar: Optional[tqdm] = None,
) -> List[Dict[str, Any]]:
    """
    Import all recalls for a make and year.

    Args:
        client: NHTSA API client.
        make: Vehicle make.
        year: Model year.
        pbar: Progress bar to update.

    Returns:
        List of recall records.
    """
    # Get models for this make/year
    models = await client.get_models_for_make_year(make, year)

    if not models:
        return []

    all_recalls = []

    for model in models:
        recalls = await client.get_recalls_for_vehicle(make, model, year)
        if recalls:
            all_recalls.extend(recalls)

        if pbar:
            pbar.set_postfix(
                make=make[:8],
                year=year,
                model=model[:12],
                recalls=len(all_recalls),
            )

    return all_recalls


async def import_all_recalls(
    makes: List[str],
    start_year: int,
    end_year: int,
    resume: bool = True,
) -> Dict[str, Any]:
    """
    Import all recalls for specified makes and years.

    Args:
        makes: List of vehicle makes.
        start_year: Start year.
        end_year: End year (inclusive).
        resume: Resume from last progress if available.

    Returns:
        Dictionary with import statistics.
    """
    client = NHTSARecallsClient()
    progress = load_progress() if resume else {}

    # Calculate total work items
    total_items = len(makes) * (end_year - start_year + 1)

    # Statistics
    stats = {
        "start_time": datetime.now(timezone.utc).isoformat(),
        "makes": makes,
        "year_range": f"{start_year}-{end_year}",
        "total_recalls": 0,
        "recalls_by_make": defaultdict(int),
        "recalls_by_year": defaultdict(int),
        "recalls_by_component": defaultdict(int),
        "unique_campaign_numbers": set(),
        "makes_with_no_data": [],
    }

    try:
        with tqdm(total=total_items, desc="Importing recalls", unit="make-year") as pbar:
            for make in makes:
                make_recalls_total = 0

                for year in range(start_year, end_year + 1):
                    # Check if already processed
                    progress_key = f"{make}_{year}"
                    if progress.get(progress_key, {}).get("completed"):
                        # Load existing data and update stats
                        year_file = DATA_DIR / make / f"{year}.json"
                        if year_file.exists():
                            year_data = load_json(year_file)
                            if year_data:
                                count = year_data.get("metadata", {}).get(
                                    "recall_count", 0
                                )
                                stats["total_recalls"] += count
                                stats["recalls_by_make"][make] += count
                                stats["recalls_by_year"][str(year)] += count
                                make_recalls_total += count
                        pbar.update(1)
                        continue

                    # Fetch recalls
                    recalls = await import_recalls_for_year(
                        client, make, year, pbar
                    )

                    # Save year data
                    year_data = {
                        "metadata": {
                            "make": make,
                            "year": year,
                            "imported_at": datetime.now(timezone.utc).isoformat(),
                            "recall_count": len(recalls),
                        },
                        "recalls": recalls,
                    }

                    year_file = DATA_DIR / make / f"{year}.json"
                    save_json(year_data, year_file)

                    # Update statistics
                    stats["total_recalls"] += len(recalls)
                    stats["recalls_by_make"][make] += len(recalls)
                    stats["recalls_by_year"][str(year)] += len(recalls)
                    make_recalls_total += len(recalls)

                    for recall in recalls:
                        component = recall.get("component", "Unknown")
                        stats["recalls_by_component"][component] += 1
                        campaign = recall.get("campaign_number")
                        if campaign:
                            stats["unique_campaign_numbers"].add(campaign)

                    # Update progress
                    progress[progress_key] = {
                        "completed": True,
                        "recall_count": len(recalls),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    save_progress(progress)

                    pbar.update(1)

                if make_recalls_total == 0:
                    stats["makes_with_no_data"].append(make)
                    logger.info(f"No recalls found for {make}")

        # Finalize statistics
        stats["end_time"] = datetime.now(timezone.utc).isoformat()
        stats["unique_campaign_numbers"] = list(stats["unique_campaign_numbers"])
        stats["recalls_by_make"] = dict(stats["recalls_by_make"])
        stats["recalls_by_year"] = dict(stats["recalls_by_year"])
        stats["recalls_by_component"] = dict(stats["recalls_by_component"])
        stats["client_stats"] = client.get_stats()

        # Save statistics
        save_json(stats, STATS_FILE)

        return stats

    finally:
        await client.close()


def generate_statistics() -> Dict[str, Any]:
    """
    Generate statistics from imported data.

    Returns:
        Dictionary with comprehensive statistics.
    """
    if not DATA_DIR.exists():
        return {"error": "No data directory found"}

    stats = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_recalls": 0,
        "total_files": 0,
        "recalls_by_make": defaultdict(int),
        "recalls_by_year": defaultdict(int),
        "recalls_by_component": defaultdict(int),
        "unique_campaign_numbers": set(),
        "oldest_recall": None,
        "newest_recall": None,
        "top_components": [],
        "top_makes": [],
    }

    # Scan all JSON files
    for make_dir in DATA_DIR.iterdir():
        if not make_dir.is_dir() or make_dir.name.startswith("."):
            continue

        make = make_dir.name

        for year_file in make_dir.glob("*.json"):
            if year_file.name in ["import_statistics.json", "import_progress.json"]:
                continue

            data = load_json(year_file)
            if not data:
                continue

            stats["total_files"] += 1
            recalls = data.get("recalls", [])
            count = len(recalls)

            stats["total_recalls"] += count
            stats["recalls_by_make"][make] += count

            year = data.get("metadata", {}).get("year")
            if year:
                stats["recalls_by_year"][str(year)] += count

            for recall in recalls:
                component = recall.get("component", "Unknown")
                stats["recalls_by_component"][component] += 1

                campaign = recall.get("campaign_number")
                if campaign:
                    stats["unique_campaign_numbers"].add(campaign)

                # Track oldest/newest recalls
                recall_date = recall.get("recall_date")
                if recall_date:
                    if stats["oldest_recall"] is None or recall_date < stats["oldest_recall"]:
                        stats["oldest_recall"] = recall_date
                    if stats["newest_recall"] is None or recall_date > stats["newest_recall"]:
                        stats["newest_recall"] = recall_date

    # Convert sets to counts
    stats["unique_campaign_count"] = len(stats["unique_campaign_numbers"])
    del stats["unique_campaign_numbers"]

    # Convert defaultdicts to regular dicts
    stats["recalls_by_make"] = dict(stats["recalls_by_make"])
    stats["recalls_by_year"] = dict(stats["recalls_by_year"])
    stats["recalls_by_component"] = dict(stats["recalls_by_component"])

    # Top components
    sorted_components = sorted(
        stats["recalls_by_component"].items(),
        key=lambda x: x[1],
        reverse=True,
    )
    stats["top_components"] = sorted_components[:20]

    # Top makes
    sorted_makes = sorted(
        stats["recalls_by_make"].items(),
        key=lambda x: x[1],
        reverse=True,
    )
    stats["top_makes"] = sorted_makes

    return stats


def print_statistics(stats: Dict[str, Any]) -> None:
    """Print formatted statistics."""
    print("\n" + "=" * 70)
    print("NHTSA RECALLS IMPORT STATISTICS")
    print("=" * 70)

    print(f"\nTotal recalls imported: {stats.get('total_recalls', 0):,}")
    print(f"Unique campaign numbers: {stats.get('unique_campaign_count', 0):,}")
    print(f"Total data files: {stats.get('total_files', 0)}")

    if stats.get("oldest_recall"):
        print(f"\nOldest recall: {stats['oldest_recall']}")
    if stats.get("newest_recall"):
        print(f"Newest recall: {stats['newest_recall']}")

    # Recalls by make
    print("\n" + "-" * 40)
    print("RECALLS BY MAKE")
    print("-" * 40)
    for make, count in sorted(
        stats.get("recalls_by_make", {}).items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        print(f"  {make:20s}: {count:>6,}")

    # Top components
    print("\n" + "-" * 40)
    print("TOP 15 COMPONENTS")
    print("-" * 40)
    top_components = stats.get("top_components", [])
    for component, count in top_components[:15]:
        # Truncate long component names
        comp_display = (
            component[:45] + "..." if len(component) > 48 else component
        )
        print(f"  {comp_display:48s}: {count:>5,}")

    # Recalls by decade
    print("\n" + "-" * 40)
    print("RECALLS BY DECADE")
    print("-" * 40)
    decades: Dict[str, int] = defaultdict(int)
    for year_str, count in stats.get("recalls_by_year", {}).items():
        decade = f"{year_str[:3]}0s"
        decades[decade] += count
    for decade in sorted(decades.keys()):
        print(f"  {decade}: {decades[decade]:>6,}")

    if stats.get("client_stats"):
        print("\n" + "-" * 40)
        print("API REQUEST STATISTICS")
        print("-" * 40)
        client_stats = stats["client_stats"]
        print(f"  Total requests: {client_stats.get('total_requests', 0):,}")
        print(f"  Errors: {client_stats.get('errors', 0)}")
        print(f"  Success rate: {client_stats.get('success_rate', 0)}%")

    if stats.get("makes_with_no_data"):
        print("\n" + "-" * 40)
        print("MAKES WITH NO DATA (limited US presence)")
        print("-" * 40)
        for make in stats["makes_with_no_data"]:
            print(f"  - {make}")

    print("\n" + "=" * 70)
    print(f"Data saved to: {DATA_DIR}")
    print("=" * 70)


# =============================================================================
# CLI Entry Point
# =============================================================================


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Import vehicle recalls from NHTSA API (1966-2024)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/import_nhtsa_recalls.py                    # Import all recalls
  python scripts/import_nhtsa_recalls.py --years 2020-2024  # Last 5 years only
  python scripts/import_nhtsa_recalls.py --makes Toyota Honda  # Specific makes
  python scripts/import_nhtsa_recalls.py --stats-only       # Show statistics only
  python scripts/import_nhtsa_recalls.py --no-resume        # Start fresh
        """,
    )

    parser.add_argument(
        "--years",
        type=str,
        default=f"{DEFAULT_START_YEAR}-{DEFAULT_END_YEAR}",
        help=f"Year range (e.g., 2020-2024). Default: {DEFAULT_START_YEAR}-{DEFAULT_END_YEAR}",
    )
    parser.add_argument(
        "--makes",
        type=str,
        nargs="+",
        default=TARGET_MAKES,
        help=f"Vehicle makes to import. Default: {', '.join(TARGET_MAKES[:5])}...",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show statistics from existing data",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignoring previous progress",
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

    # Parse year range
    try:
        year_parts = args.years.split("-")
        start_year = int(year_parts[0])
        end_year = int(year_parts[1]) if len(year_parts) > 1 else start_year
    except (ValueError, IndexError):
        logger.error(f"Invalid year range: {args.years}")
        sys.exit(1)

    # Validate year range
    if start_year < 1966:
        logger.warning("Start year before 1966, setting to 1966")
        start_year = 1966
    if end_year > 2024:
        logger.warning("End year after 2024, setting to 2024")
        end_year = 2024

    # Create output directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Stats only mode
    if args.stats_only:
        stats = generate_statistics()
        print_statistics(stats)
        save_json(stats, STATS_FILE)
        return

    # Print configuration
    print("\n" + "=" * 70)
    print("NHTSA RECALLS IMPORT")
    print("=" * 70)
    print(f"Makes: {', '.join(args.makes[:5])}{'...' if len(args.makes) > 5 else ''}")
    print(f"Year range: {start_year}-{end_year} ({end_year - start_year + 1} years)")
    print(f"Resume from progress: {not args.no_resume}")
    print(f"Output directory: {DATA_DIR}")
    print("=" * 70 + "\n")

    # Run import
    try:
        stats = await import_all_recalls(
            makes=args.makes,
            start_year=start_year,
            end_year=end_year,
            resume=not args.no_resume,
        )

        # Generate and print comprehensive statistics
        full_stats = generate_statistics()
        full_stats["import_stats"] = stats
        print_statistics(full_stats)
        save_json(full_stats, STATS_FILE)

        logger.info("Import completed successfully!")

    except KeyboardInterrupt:
        logger.info("\nImport interrupted by user. Progress saved.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Import failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
