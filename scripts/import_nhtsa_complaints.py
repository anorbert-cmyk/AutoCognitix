#!/usr/bin/env python3
"""
NHTSA Complaints Import Script for AutoCognitix.

Downloads vehicle complaint data from NHTSA API for specified makes
and year range, saves to JSON files organized by year.

Features:
- Async HTTP requests with aiohttp
- Rate limiting (max 10 req/sec)
- Progress bar (tqdm)
- Retry logic with exponential backoff
- Saves data to data/nhtsa/complaints/
- Generates summary statistics

NHTSA API Documentation:
- Endpoint: https://api.nhtsa.gov/complaints/complaintsByVehicle
- Parameters: make, model, modelYear
- No authentication required

Usage:
    python scripts/import_nhtsa_complaints.py                  # All defaults
    python scripts/import_nhtsa_complaints.py --years 2020-2024
    python scripts/import_nhtsa_complaints.py --dry-run        # Preview only
    python scripts/import_nhtsa_complaints.py -v               # Verbose mode
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
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


# =============================================================================
# Configuration
# =============================================================================

# NHTSA API endpoints
NHTSA_BASE_URL = "https://api.nhtsa.gov"
COMPLAINTS_ENDPOINT = f"{NHTSA_BASE_URL}/complaints/complaintsByVehicle"
VPIC_BASE_URL = "https://vpic.nhtsa.dot.gov/api/vehicles"

# Output paths
DATA_DIR = PROJECT_ROOT / "data" / "nhtsa" / "complaints"

# Rate limiting: 10 requests per second
MAX_REQUESTS_PER_SECOND = 10
REQUEST_INTERVAL = 1.0 / MAX_REQUESTS_PER_SECOND  # 0.1 seconds

# Retry configuration
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0  # Exponential backoff base

# Year range
DEFAULT_START_YEAR = 2014
DEFAULT_END_YEAR = 2024

# Top vehicle makes (European + American + Japanese + Korean)
TOP_MAKES = [
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
    # Note: Opel and Skoda are rarely in NHTSA (US market)
]

# Popular models per make (fallback if API model list fails)
POPULAR_MODELS: Dict[str, List[str]] = {
    "Toyota": ["Camry", "Corolla", "RAV4", "Highlander", "Tacoma", "Prius"],
    "Honda": ["Civic", "Accord", "CR-V", "Pilot", "Odyssey", "HR-V"],
    "Ford": ["F-150", "Escape", "Explorer", "Mustang", "Fusion", "Edge"],
    "Chevrolet": ["Silverado", "Equinox", "Malibu", "Traverse", "Tahoe", "Cruze"],
    "Nissan": ["Altima", "Rogue", "Sentra", "Maxima", "Pathfinder", "Murano"],
    "BMW": ["3 Series", "5 Series", "X3", "X5", "7 Series", "X1"],
    "Mercedes-Benz": ["C-Class", "E-Class", "GLE", "GLC", "S-Class", "A-Class"],
    "Volkswagen": ["Jetta", "Passat", "Tiguan", "Atlas", "Golf", "Beetle"],
    "Audi": ["A4", "A6", "Q5", "Q7", "A3", "Q3"],
    "Hyundai": ["Elantra", "Sonata", "Tucson", "Santa Fe", "Kona", "Accent"],
    "Kia": ["Optima", "Sorento", "Sportage", "Soul", "Forte", "Telluride"],
    "Mazda": ["Mazda3", "Mazda6", "CX-5", "CX-9", "MX-5 Miata", "CX-3"],
    "Subaru": ["Outback", "Forester", "Crosstrek", "Impreza", "Legacy", "Ascent"],
}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class Complaint:
    """Represents a single NHTSA complaint record."""

    odi_number: str
    manufacturer: str
    make: str
    model: str
    model_year: int
    component: str
    summary: str
    crash: bool = False
    fire: bool = False
    injuries: int = 0
    deaths: int = 0
    complaint_date: Optional[str] = None
    date_of_incident: Optional[str] = None
    mileage: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "odi_number": self.odi_number,
            "manufacturer": self.manufacturer,
            "make": self.make,
            "model": self.model,
            "model_year": self.model_year,
            "component": self.component,
            "summary": self.summary,
            "crash": self.crash,
            "fire": self.fire,
            "injuries": self.injuries,
            "deaths": self.deaths,
            "complaint_date": self.complaint_date,
            "date_of_incident": self.date_of_incident,
            "mileage": self.mileage,
        }


@dataclass
class ImportStats:
    """Statistics for the import process."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_complaints: int = 0
    complaints_by_make: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    complaints_by_year: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    complaints_by_component: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    crashes: int = 0
    fires: int = 0
    injuries: int = 0
    deaths: int = 0
    start_time: float = 0.0
    end_time: float = 0.0

    def add_complaint(self, complaint: Complaint) -> None:
        """Update statistics with a new complaint."""
        self.total_complaints += 1
        self.complaints_by_make[complaint.make] += 1
        self.complaints_by_year[complaint.model_year] += 1
        self.complaints_by_component[complaint.component] += 1

        if complaint.crash:
            self.crashes += 1
        if complaint.fire:
            self.fires += 1
        self.injuries += complaint.injuries
        self.deaths += complaint.deaths

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        elapsed = self.end_time - self.start_time if self.end_time else 0
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "total_complaints": self.total_complaints,
            "complaints_by_make": dict(self.complaints_by_make),
            "complaints_by_year": dict(self.complaints_by_year),
            "top_components": dict(
                sorted(
                    self.complaints_by_component.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:20]
            ),
            "safety_metrics": {
                "crashes": self.crashes,
                "fires": self.fires,
                "injuries": self.injuries,
                "deaths": self.deaths,
            },
            "elapsed_seconds": round(elapsed, 2),
            "requests_per_second": (
                round(self.total_requests / elapsed, 2) if elapsed > 0 else 0
            ),
        }


# =============================================================================
# Rate Limiter
# =============================================================================


class RateLimiter:
    """Token bucket rate limiter for async requests."""

    def __init__(self, rate: float = MAX_REQUESTS_PER_SECOND):
        """
        Initialize rate limiter.

        Args:
            rate: Maximum requests per second.
        """
        self.rate = rate
        self.interval = 1.0 / rate
        self._last_request = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a request can be made within rate limits."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_request
            if elapsed < self.interval:
                await asyncio.sleep(self.interval - elapsed)
            self._last_request = time.monotonic()


# =============================================================================
# NHTSA API Client
# =============================================================================


class NHTSAComplaintsClient:
    """Async NHTSA API client with rate limiting and retry logic."""

    def __init__(
        self,
        rate_limiter: RateLimiter,
        max_retries: int = MAX_RETRIES,
    ):
        """
        Initialize the client.

        Args:
            rate_limiter: Rate limiter instance.
            max_retries: Maximum retry attempts on failure.
        """
        self.rate_limiter = rate_limiter
        self.max_retries = max_retries
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            connector = aiohttp.TCPConnector(limit=20, limit_per_host=10)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
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
        Make an API request with rate limiting and retry logic.

        Args:
            url: Request URL.
            params: Query parameters.

        Returns:
            JSON response or None on failure.
        """
        session = await self._get_session()

        for attempt in range(self.max_retries):
            await self.rate_limiter.acquire()

            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        return await response.json()

                    if response.status == 429:
                        # Rate limited - exponential backoff
                        wait_time = RETRY_BACKOFF_BASE ** (attempt + 1)
                        logger.warning(
                            f"Rate limited (429). Waiting {wait_time}s before retry..."
                        )
                        await asyncio.sleep(wait_time)
                        continue

                    if response.status >= 500:
                        # Server error - retry
                        wait_time = RETRY_BACKOFF_BASE**attempt
                        logger.warning(
                            f"Server error {response.status}. "
                            f"Retry {attempt + 1}/{self.max_retries} in {wait_time}s"
                        )
                        await asyncio.sleep(wait_time)
                        continue

                    # Client error (4xx except 429) - don't retry
                    logger.debug(f"Client error {response.status} for {url}")
                    return None

            except asyncio.TimeoutError:
                wait_time = RETRY_BACKOFF_BASE**attempt
                logger.warning(
                    f"Timeout. Retry {attempt + 1}/{self.max_retries} in {wait_time}s"
                )
                await asyncio.sleep(wait_time)

            except aiohttp.ClientError as e:
                wait_time = RETRY_BACKOFF_BASE**attempt
                logger.warning(
                    f"Client error: {e}. "
                    f"Retry {attempt + 1}/{self.max_retries} in {wait_time}s"
                )
                await asyncio.sleep(wait_time)

        logger.error(f"Failed after {self.max_retries} retries: {url}")
        return None

    async def get_models_for_make_year(self, make: str, year: int) -> List[str]:
        """
        Get available models for a make and year from VPIC API.

        Args:
            make: Vehicle make.
            year: Model year.

        Returns:
            List of model names.
        """
        url = f"{VPIC_BASE_URL}/GetModelsForMakeYear/make/{make}/modelyear/{year}"
        params = {"format": "json"}

        data = await self._make_request(url, params)
        if not data:
            # Fall back to popular models
            return POPULAR_MODELS.get(make, [])

        results = data.get("Results", [])
        models = [r.get("Model_Name", "") for r in results if r.get("Model_Name")]

        # If no models found, use fallback
        if not models:
            return POPULAR_MODELS.get(make, [])

        return models

    async def get_complaints(
        self,
        make: str,
        model: str,
        year: int,
    ) -> List[Complaint]:
        """
        Fetch complaints for a specific vehicle.

        Args:
            make: Vehicle make.
            model: Vehicle model.
            year: Model year.

        Returns:
            List of Complaint objects.
        """
        params = {
            "make": make,
            "model": model,
            "modelYear": year,
        }

        data = await self._make_request(COMPLAINTS_ENDPOINT, params)
        if not data:
            return []

        results = data.get("results", [])
        complaints = []

        for item in results:
            try:
                complaint = Complaint(
                    odi_number=str(item.get("odiNumber", "")),
                    manufacturer=item.get("manufacturer", make),
                    make=make,
                    model=model,
                    model_year=year,
                    component=item.get("components", "Unknown"),
                    summary=item.get("summary", "") or "",
                    crash=item.get("crash", "N") == "Y",
                    fire=item.get("fire", "N") == "Y",
                    injuries=int(item.get("numberOfInjuries") or 0),
                    deaths=int(item.get("numberOfDeaths") or 0),
                    complaint_date=item.get("dateComplaintFiled"),
                    date_of_incident=item.get("dateOfIncident"),
                    mileage=(
                        int(item.get("mileage"))
                        if item.get("mileage")
                        and str(item.get("mileage")).isdigit()
                        else None
                    ),
                )
                complaints.append(complaint)
            except (ValueError, TypeError, KeyError) as e:
                logger.debug(f"Error parsing complaint: {e}")
                continue

        return complaints

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()


# =============================================================================
# Import Functions
# =============================================================================


async def import_complaints(
    makes: List[str],
    start_year: int,
    end_year: int,
    output_dir: Path,
    max_models_per_make: int = 10,
    dry_run: bool = False,
) -> ImportStats:
    """
    Import complaints for specified makes and years.

    Args:
        makes: List of vehicle makes.
        start_year: Start year (inclusive).
        end_year: End year (inclusive).
        output_dir: Directory to save JSON files.
        max_models_per_make: Maximum models to fetch per make/year.
        dry_run: If True, don't save files.

    Returns:
        ImportStats with results.
    """
    stats = ImportStats()
    stats.start_time = time.monotonic()

    rate_limiter = RateLimiter(rate=MAX_REQUESTS_PER_SECOND)
    client = NHTSAComplaintsClient(rate_limiter)

    # Create output directory
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate total work items for progress bar
    years = list(range(start_year, end_year + 1))
    total_make_years = len(makes) * len(years)

    # Collect all complaints by year for saving
    complaints_by_year: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    try:
        with tqdm(
            total=total_make_years,
            desc="Importing complaints",
            unit="make/year",
        ) as pbar:
            for make in makes:
                for year in years:
                    pbar.set_postfix(make=make, year=year)

                    # Get models for this make/year
                    models = await client.get_models_for_make_year(make, year)
                    models = models[:max_models_per_make]

                    for model in models:
                        stats.total_requests += 1

                        complaints = await client.get_complaints(make, model, year)

                        if complaints:
                            stats.successful_requests += 1

                            for complaint in complaints:
                                stats.add_complaint(complaint)
                                complaints_by_year[year].append(complaint.to_dict())

                            logger.debug(
                                f"{make} {model} {year}: {len(complaints)} complaints"
                            )
                        else:
                            # Could be no data or error
                            stats.failed_requests += 1

                    pbar.update(1)

        # Save data to files
        if not dry_run:
            for year, year_complaints in complaints_by_year.items():
                if year_complaints:
                    output_file = output_dir / f"complaints_{year}.json"
                    data = {
                        "metadata": {
                            "year": year,
                            "imported_at": datetime.now(timezone.utc).isoformat(),
                            "makes": makes,
                            "total_complaints": len(year_complaints),
                        },
                        "complaints": year_complaints,
                    }
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    logger.info(f"Saved {len(year_complaints)} complaints to {output_file}")

            # Save summary statistics
            stats.end_time = time.monotonic()
            stats_file = output_dir / "import_stats.json"
            with open(stats_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "metadata": {
                            "imported_at": datetime.now(timezone.utc).isoformat(),
                            "year_range": f"{start_year}-{end_year}",
                            "makes": makes,
                        },
                        "statistics": stats.to_dict(),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            logger.info(f"Saved statistics to {stats_file}")

    finally:
        await client.close()

    stats.end_time = time.monotonic()
    return stats


def print_summary(stats: ImportStats) -> None:
    """Print import summary to console."""
    elapsed = stats.end_time - stats.start_time

    print("\n" + "=" * 70)
    print("NHTSA COMPLAINTS IMPORT SUMMARY")
    print("=" * 70)

    print(f"\n{'Requests:':<25}")
    print(f"  Total:                 {stats.total_requests}")
    print(f"  Successful:            {stats.successful_requests}")
    print(f"  Failed:                {stats.failed_requests}")

    print(f"\n{'Complaints:':<25}")
    print(f"  Total:                 {stats.total_complaints:,}")

    print(f"\n{'By Make:':<25}")
    for make, count in sorted(
        stats.complaints_by_make.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {make:<20} {count:>8,}")

    print(f"\n{'By Year:':<25}")
    for year, count in sorted(stats.complaints_by_year.items()):
        print(f"  {year:<20} {count:>8,}")

    print(f"\n{'Safety Metrics:':<25}")
    print(f"  Crashes:               {stats.crashes:,}")
    print(f"  Fires:                 {stats.fires:,}")
    print(f"  Injuries:              {stats.injuries:,}")
    print(f"  Deaths:                {stats.deaths:,}")

    print(f"\n{'Top Components:':<25}")
    top_components = sorted(
        stats.complaints_by_component.items(), key=lambda x: x[1], reverse=True
    )[:10]
    for component, count in top_components:
        comp_name = component[:35] if len(component) > 35 else component
        print(f"  {comp_name:<35} {count:>8,}")

    print(f"\n{'Performance:':<25}")
    print(f"  Elapsed time:          {elapsed:.1f} seconds")
    if elapsed > 0:
        print(f"  Requests/second:       {stats.total_requests / elapsed:.1f}")

    print("=" * 70)


# =============================================================================
# CLI Entry Point
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Import NHTSA complaint data for vehicle diagnostics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/import_nhtsa_complaints.py
    python scripts/import_nhtsa_complaints.py --years 2020-2024
    python scripts/import_nhtsa_complaints.py --makes Toyota Honda Ford
    python scripts/import_nhtsa_complaints.py --dry-run
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
        default=TOP_MAKES,
        help=f"Vehicle makes to fetch. Default: {', '.join(TOP_MAKES[:5])}...",
    )

    parser.add_argument(
        "--max-models",
        type=int,
        default=10,
        help="Maximum models per make/year (default: 10)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DATA_DIR),
        help=f"Output directory. Default: {DATA_DIR}",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview mode - don't save files",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


async def main() -> int:
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse year range
    try:
        year_parts = args.years.split("-")
        start_year = int(year_parts[0])
        end_year = int(year_parts[1]) if len(year_parts) > 1 else start_year
    except (ValueError, IndexError):
        logger.error(f"Invalid year range: {args.years}")
        return 1

    # Validate years
    current_year = datetime.now().year
    if start_year < 1995 or end_year > current_year + 1:
        logger.error(f"Invalid year range: {start_year}-{end_year}")
        return 1

    output_dir = Path(args.output_dir)

    logger.info("=" * 60)
    logger.info("NHTSA Complaints Import")
    logger.info("=" * 60)
    logger.info(f"Makes: {', '.join(args.makes)}")
    logger.info(f"Years: {start_year}-{end_year}")
    logger.info(f"Max models/make: {args.max_models}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info("=" * 60)

    try:
        stats = await import_complaints(
            makes=args.makes,
            start_year=start_year,
            end_year=end_year,
            output_dir=output_dir,
            max_models_per_make=args.max_models,
            dry_run=args.dry_run,
        )

        print_summary(stats)

        if stats.total_complaints > 0:
            logger.info("Import completed successfully!")
            return 0
        else:
            logger.warning("No complaints imported")
            return 1

    except KeyboardInterrupt:
        logger.info("\nImport cancelled by user")
        return 130

    except Exception as e:
        logger.error(f"Import failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
