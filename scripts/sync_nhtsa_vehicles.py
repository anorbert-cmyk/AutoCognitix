#!/usr/bin/env python3
"""
Comprehensive NHTSA Vehicle Data Sync Script for AutoCognitix.

Downloads and stores complete vehicle information from NHTSA APIs:
- All vehicle makes (~10,000+)
- Models for top manufacturers
- Recalls by make/model
- Complaints by make/model

Features:
- Resume capability (save progress to file and database)
- Rate limiting (respects NHTSA API limits: 10 req/sec)
- Progress tracking with tqdm
- Error handling and automatic retries
- Logging to file and console
- Command line arguments for selective sync

NHTSA API Documentation:
- VPIC API: https://vpic.nhtsa.dot.gov/api/
- Recalls API: https://api.nhtsa.gov/recalls
- Complaints API: https://api.nhtsa.gov/complaints

Usage:
    python scripts/sync_nhtsa_vehicles.py                    # Full sync
    python scripts/sync_nhtsa_vehicles.py --makes-only       # Only sync makes
    python scripts/sync_nhtsa_vehicles.py --models-only      # Only sync models
    python scripts/sync_nhtsa_vehicles.py --recalls          # Sync recalls
    python scripts/sync_nhtsa_vehicles.py --complaints       # Sync complaints
    python scripts/sync_nhtsa_vehicles.py --limit 100        # Limit items
    python scripts/sync_nhtsa_vehicles.py --resume           # Resume from progress
    python scripts/sync_nhtsa_vehicles.py --verbose          # Verbose logging
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Try to import database dependencies
try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.exc import IntegrityError, OperationalError

    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False

# =============================================================================
# Configuration
# =============================================================================

# NHTSA API endpoints
VPIC_BASE_URL = "https://vpic.nhtsa.dot.gov/api/vehicles"
NHTSA_API_BASE_URL = "https://api.nhtsa.gov"

# API Endpoints
GET_ALL_MAKES_URL = f"{VPIC_BASE_URL}/GetAllMakes"
GET_MODELS_FOR_MAKE_URL = f"{VPIC_BASE_URL}/GetModelsForMake"
GET_MODELS_FOR_MAKE_YEAR_URL = f"{VPIC_BASE_URL}/GetModelsForMakeYear"
RECALLS_ENDPOINT = f"{NHTSA_API_BASE_URL}/recalls/recallsByVehicle"
COMPLAINTS_ENDPOINT = f"{NHTSA_API_BASE_URL}/complaints/complaintsByVehicle"

# Rate limiting: NHTSA allows ~10 requests per second
MAX_REQUESTS_PER_SECOND = 8  # Conservative to avoid rate limiting
REQUEST_INTERVAL = 1.0 / MAX_REQUESTS_PER_SECOND

# Output paths
DATA_DIR = PROJECT_ROOT / "data" / "nhtsa"
PROGRESS_FILE = DATA_DIR / "sync_progress.json"
MAKES_FILE = DATA_DIR / "all_makes.json"
MODELS_FILE = DATA_DIR / "all_models.json"
LOG_FILE = DATA_DIR / "sync_nhtsa_vehicles.log"

# Popular makes to prioritize for model/recalls/complaints sync
TOP_MAKES = [
    # European
    "Volkswagen", "BMW", "Mercedes-Benz", "Audi", "Volvo", "Porsche",
    "Land Rover", "Jaguar", "Mini", "Fiat", "Alfa Romeo", "Peugeot",
    "Renault", "Citroen", "Opel", "Skoda", "Seat",
    # American
    "Ford", "Chevrolet", "GMC", "Jeep", "Ram", "Dodge", "Chrysler",
    "Cadillac", "Buick", "Lincoln", "Tesla",
    # Japanese
    "Toyota", "Honda", "Nissan", "Mazda", "Subaru", "Mitsubishi",
    "Lexus", "Acura", "Infiniti", "Suzuki", "Isuzu",
    # Korean
    "Hyundai", "Kia", "Genesis",
]

# Default year range for models/recalls/complaints
DEFAULT_START_YEAR = 2010
DEFAULT_END_YEAR = 2025

# DTC code patterns for extraction from recalls/complaints
import re

DTC_PATTERN = re.compile(r"\b([PCBU][0-9A-Fa-f]{4})\b", re.IGNORECASE)
EXTENDED_DTC_PATTERNS = [
    re.compile(r"DTC\s*[:\-]?\s*([PCBU][0-9A-Fa-f]{4})", re.IGNORECASE),
    re.compile(r"code\s*[:\-]?\s*([PCBU][0-9A-Fa-f]{4})", re.IGNORECASE),
    re.compile(r"trouble\s+code\s*[:\-]?\s*([PCBU][0-9A-Fa-f]{4})", re.IGNORECASE),
]


# =============================================================================
# Logging Configuration
# =============================================================================


def setup_logging(verbose: bool = False, log_file: Optional[Path] = None) -> logging.Logger:
    """Configure logging with file and console handlers."""
    logger = logging.getLogger("nhtsa_vehicle_sync")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_format = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


logger = setup_logging(log_file=LOG_FILE)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class SyncProgress:
    """Track sync progress for resume capability."""

    makes_synced: bool = False
    makes_count: int = 0
    models_synced_makes: List[str] = field(default_factory=list)
    recalls_synced_vehicles: List[str] = field(default_factory=list)  # "make:model:year"
    complaints_synced_vehicles: List[str] = field(default_factory=list)
    last_updated: str = ""
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "makes_synced": self.makes_synced,
            "makes_count": self.makes_count,
            "models_synced_makes": self.models_synced_makes,
            "recalls_synced_vehicles": self.recalls_synced_vehicles,
            "complaints_synced_vehicles": self.complaints_synced_vehicles,
            "last_updated": self.last_updated,
            "errors": self.errors[-100:],  # Keep last 100 errors
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SyncProgress":
        return cls(
            makes_synced=data.get("makes_synced", False),
            makes_count=data.get("makes_count", 0),
            models_synced_makes=data.get("models_synced_makes", []),
            recalls_synced_vehicles=data.get("recalls_synced_vehicles", []),
            complaints_synced_vehicles=data.get("complaints_synced_vehicles", []),
            last_updated=data.get("last_updated", ""),
            errors=data.get("errors", []),
        )


@dataclass
class SyncStats:
    """Statistics for sync operation."""

    makes_fetched: int = 0
    makes_saved: int = 0
    models_fetched: int = 0
    models_saved: int = 0
    recalls_fetched: int = 0
    recalls_saved: int = 0
    complaints_fetched: int = 0
    complaints_saved: int = 0
    dtc_codes_extracted: int = 0
    api_requests: int = 0
    api_errors: int = 0
    start_time: float = field(default_factory=time.time)

    def elapsed_time(self) -> float:
        return time.time() - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "makes": {"fetched": self.makes_fetched, "saved": self.makes_saved},
            "models": {"fetched": self.models_fetched, "saved": self.models_saved},
            "recalls": {"fetched": self.recalls_fetched, "saved": self.recalls_saved},
            "complaints": {
                "fetched": self.complaints_fetched,
                "saved": self.complaints_saved,
            },
            "dtc_codes_extracted": self.dtc_codes_extracted,
            "api": {"requests": self.api_requests, "errors": self.api_errors},
            "elapsed_seconds": round(self.elapsed_time(), 2),
        }


@dataclass
class VehicleMake:
    """Vehicle make data from NHTSA."""

    make_id: int
    make_name: str
    country: Optional[str] = None

    @property
    def normalized_id(self) -> str:
        """Generate a normalized ID for database storage."""
        return self.make_name.lower().replace(" ", "_").replace("-", "_")[:50]


@dataclass
class VehicleModel:
    """Vehicle model data from NHTSA."""

    model_id: int
    model_name: str
    make_id: int
    make_name: str
    year: Optional[int] = None

    @property
    def normalized_id(self) -> str:
        """Generate a normalized ID for database storage."""
        make_part = self.make_name.lower().replace(" ", "_").replace("-", "_")[:25]
        model_part = self.model_name.lower().replace(" ", "_").replace("-", "_")[:25]
        return f"{make_part}_{model_part}"[:50]


@dataclass
class RecallRecord:
    """Recall record from NHTSA API."""

    campaign_number: str
    nhtsa_id: Optional[str]
    manufacturer: str
    make: str
    model: str
    model_year: int
    recall_date: Optional[date]
    component: Optional[str]
    summary: Optional[str]
    consequence: Optional[str]
    remedy: Optional[str]
    notes: Optional[str]
    extracted_dtc_codes: List[str]
    raw_response: Dict[str, Any]


@dataclass
class ComplaintRecord:
    """Complaint record from NHTSA API."""

    odi_number: str
    manufacturer: str
    make: str
    model: str
    model_year: int
    crash: bool
    fire: bool
    injuries: int
    deaths: int
    complaint_date: Optional[date]
    date_of_incident: Optional[date]
    components: Optional[str]
    summary: Optional[str]
    extracted_dtc_codes: List[str]
    raw_response: Dict[str, Any]


# =============================================================================
# Utility Functions
# =============================================================================


def extract_dtc_codes(text: str) -> List[str]:
    """Extract DTC codes from text content."""
    if not text:
        return []

    codes: Set[str] = set()

    # Primary pattern
    for match in DTC_PATTERN.finditer(text):
        codes.add(match.group(1).upper())

    # Extended patterns
    for pattern in EXTENDED_DTC_PATTERNS:
        for match in pattern.finditer(text):
            codes.add(match.group(1).upper())

    return sorted(codes)


def parse_date(date_str: Optional[str]) -> Optional[date]:
    """Parse date string from NHTSA API."""
    if not date_str:
        return None

    formats = ["%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d", "%d/%m/%Y"]

    for fmt in formats:
        try:
            return datetime.strptime(date_str[:10], fmt).date()
        except (ValueError, TypeError):
            continue

    return None


def save_json(data: Any, file_path: Path) -> None:
    """Save data to JSON file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def load_json(file_path: Path) -> Optional[Any]:
    """Load data from JSON file."""
    if not file_path.exists():
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Rate Limiter
# =============================================================================


class RateLimiter:
    """Token bucket rate limiter for API requests."""

    def __init__(self, rate: float = MAX_REQUESTS_PER_SECOND):
        self.rate = rate
        self.interval = 1.0 / rate
        self._lock = asyncio.Lock()
        self._last_request = 0.0

    async def acquire(self) -> None:
        """Wait for rate limit slot."""
        async with self._lock:
            now = time.time()
            elapsed = now - self._last_request
            if elapsed < self.interval:
                await asyncio.sleep(self.interval - elapsed)
            self._last_request = time.time()


# =============================================================================
# NHTSA API Client
# =============================================================================


class NHTSAVehicleClient:
    """Async NHTSA API client for vehicle data."""

    def __init__(self, rate_limiter: RateLimiter, stats: SyncStats):
        self.rate_limiter = rate_limiter
        self.stats = stats
        self._client: Optional[httpx.AsyncClient] = None
        self._retry_count = 3
        self._retry_delay = 5.0

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0),
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
                headers={
                    "User-Agent": "AutoCognitix/2.0 (Vehicle Diagnostic Platform)",
                    "Accept": "application/json",
                },
            )
        return self._client

    async def _request(
        self, url: str, params: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Make rate-limited API request with retry logic."""
        await self.rate_limiter.acquire()
        self.stats.api_requests += 1

        client = await self._get_client()

        for attempt in range(self._retry_count):
            try:
                response = await client.get(url, params=params)

                if response.status_code == 429:
                    # Rate limited - wait and retry
                    retry_after = int(response.headers.get("Retry-After", 60))
                    logger.warning(f"Rate limited, waiting {retry_after}s...")
                    await asyncio.sleep(retry_after)
                    continue

                if response.status_code >= 500:
                    # Server error - retry
                    logger.warning(
                        f"Server error {response.status_code}, retrying ({attempt + 1}/{self._retry_count})..."
                    )
                    await asyncio.sleep(self._retry_delay * (attempt + 1))
                    continue

                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP {e.response.status_code}: {url}")
                self.stats.api_errors += 1
                if attempt < self._retry_count - 1:
                    await asyncio.sleep(self._retry_delay)
                    continue
                return None
            except httpx.TimeoutException:
                logger.error(f"Timeout: {url}")
                self.stats.api_errors += 1
                if attempt < self._retry_count - 1:
                    await asyncio.sleep(self._retry_delay)
                    continue
                return None
            except Exception as e:
                logger.error(f"Request error: {e}")
                self.stats.api_errors += 1
                return None

        return None

    async def get_all_makes(self) -> List[VehicleMake]:
        """Fetch all vehicle makes from NHTSA VPIC API."""
        logger.info("Fetching all vehicle makes from NHTSA...")

        data = await self._request(GET_ALL_MAKES_URL, {"format": "json"})
        if not data:
            logger.error("Failed to fetch makes")
            return []

        results = data.get("Results", [])
        makes = []

        for item in results:
            make_id = item.get("Make_ID")
            make_name = item.get("Make_Name")

            if make_id and make_name:
                makes.append(
                    VehicleMake(make_id=make_id, make_name=make_name.strip())
                )

        self.stats.makes_fetched = len(makes)
        logger.info(f"Fetched {len(makes)} vehicle makes")
        return makes

    async def get_models_for_make(self, make_name: str) -> List[VehicleModel]:
        """Fetch all models for a specific make."""
        url = f"{GET_MODELS_FOR_MAKE_URL}/{make_name}"
        data = await self._request(url, {"format": "json"})

        if not data:
            return []

        results = data.get("Results", [])
        models = []

        for item in results:
            model_id = item.get("Model_ID")
            model_name = item.get("Model_Name")
            make_id = item.get("Make_ID")

            if model_id and model_name:
                models.append(
                    VehicleModel(
                        model_id=model_id,
                        model_name=model_name.strip(),
                        make_id=make_id,
                        make_name=make_name,
                    )
                )

        self.stats.models_fetched += len(models)
        return models

    async def get_models_for_make_year(
        self, make_name: str, year: int
    ) -> List[VehicleModel]:
        """Fetch models for a specific make and year."""
        url = f"{GET_MODELS_FOR_MAKE_YEAR_URL}/make/{make_name}/modelyear/{year}"
        data = await self._request(url, {"format": "json"})

        if not data:
            return []

        results = data.get("Results", [])
        models = []

        for item in results:
            model_id = item.get("Model_ID")
            model_name = item.get("Model_Name")
            make_id = item.get("Make_ID")

            if model_id and model_name:
                models.append(
                    VehicleModel(
                        model_id=model_id,
                        model_name=model_name.strip(),
                        make_id=make_id,
                        make_name=make_name,
                        year=year,
                    )
                )

        self.stats.models_fetched += len(models)
        return models

    async def get_recalls(
        self, make: str, model: str, year: int
    ) -> List[RecallRecord]:
        """Fetch recalls for a specific vehicle."""
        data = await self._request(
            RECALLS_ENDPOINT,
            {"make": make, "model": model, "modelYear": year},
        )

        if not data:
            return []

        results = data.get("results", [])
        records = []

        for item in results:
            summary = item.get("Summary", "") or ""
            consequence = item.get("Consequence", "") or ""
            remedy = item.get("Remedy", "") or ""
            notes = item.get("Notes", "") or ""

            full_text = f"{summary} {consequence} {remedy} {notes}"
            dtc_codes = extract_dtc_codes(full_text)

            campaign_number = item.get("NHTSACampaignNumber", "")
            if not campaign_number:
                continue

            record = RecallRecord(
                campaign_number=campaign_number,
                nhtsa_id=item.get("NHTSAActionNumber"),
                manufacturer=item.get("Manufacturer", make),
                make=make,
                model=model,
                model_year=year,
                recall_date=parse_date(item.get("ReportReceivedDate")),
                component=item.get("Component"),
                summary=summary or None,
                consequence=consequence or None,
                remedy=remedy or None,
                notes=notes or None,
                extracted_dtc_codes=dtc_codes,
                raw_response=item,
            )
            records.append(record)
            self.stats.recalls_fetched += 1

            if dtc_codes:
                self.stats.dtc_codes_extracted += len(dtc_codes)

        return records

    async def get_complaints(
        self, make: str, model: str, year: int
    ) -> List[ComplaintRecord]:
        """Fetch complaints for a specific vehicle."""
        data = await self._request(
            COMPLAINTS_ENDPOINT,
            {"make": make, "model": model, "modelYear": year},
        )

        if not data:
            return []

        results = data.get("results", [])
        records = []

        for item in results:
            summary = item.get("summary", "") or ""
            components = item.get("components", "") or ""

            full_text = f"{summary} {components}"
            dtc_codes = extract_dtc_codes(full_text)

            odi_number = item.get("odiNumber")
            if not odi_number:
                continue

            record = ComplaintRecord(
                odi_number=str(odi_number),
                manufacturer=item.get("manufacturer", make),
                make=make,
                model=model,
                model_year=year,
                crash=item.get("crash", "N") == "Y",
                fire=item.get("fire", "N") == "Y",
                injuries=int(item.get("numberOfInjuries") or 0),
                deaths=int(item.get("numberOfDeaths") or 0),
                complaint_date=parse_date(item.get("dateComplaintFiled")),
                date_of_incident=parse_date(item.get("dateOfIncident")),
                components=components or None,
                summary=summary or None,
                extracted_dtc_codes=dtc_codes,
                raw_response=item,
            )
            records.append(record)
            self.stats.complaints_fetched += 1

            if dtc_codes:
                self.stats.dtc_codes_extracted += len(dtc_codes)

        return records

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


# =============================================================================
# Database Manager
# =============================================================================


class DatabaseManager:
    """Manages PostgreSQL database operations."""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self._engine = None

    def _get_engine(self):
        """Get synchronous engine for database operations."""
        if self._engine is None:
            # Convert async URL to sync for simple operations
            sync_url = self.database_url.replace("+asyncpg", "")
            self._engine = create_engine(sync_url, pool_pre_ping=True)
        return self._engine

    def save_makes(self, makes: List[VehicleMake], stats: SyncStats) -> int:
        """Save vehicle makes to database."""
        if not makes:
            return 0

        engine = self._get_engine()
        saved_count = 0

        with engine.begin() as conn:
            for make in makes:
                try:
                    query = text(
                        """
                        INSERT INTO vehicle_makes (id, name)
                        VALUES (:id, :name)
                        ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name
                    """
                    )

                    conn.execute(
                        query,
                        {
                            "id": make.normalized_id,
                            "name": make.make_name,
                        },
                    )
                    saved_count += 1
                except Exception as e:
                    logger.warning(f"Failed to save make {make.make_name}: {e}")

        stats.makes_saved = saved_count
        return saved_count

    def save_models(
        self, models: List[VehicleModel], stats: SyncStats
    ) -> int:
        """Save vehicle models to database."""
        if not models:
            return 0

        engine = self._get_engine()
        saved_count = 0

        with engine.begin() as conn:
            for model in models:
                try:
                    # First ensure make exists
                    make_id = model.make_name.lower().replace(" ", "_").replace("-", "_")[:50]

                    # Check if make exists, insert if not
                    check_make = text(
                        "SELECT id FROM vehicle_makes WHERE id = :id"
                    )
                    result = conn.execute(check_make, {"id": make_id})
                    if not result.fetchone():
                        insert_make = text(
                            """
                            INSERT INTO vehicle_makes (id, name)
                            VALUES (:id, :name)
                            ON CONFLICT (id) DO NOTHING
                        """
                        )
                        conn.execute(
                            insert_make,
                            {"id": make_id, "name": model.make_name},
                        )

                    # Now insert model
                    query = text(
                        """
                        INSERT INTO vehicle_models (id, name, make_id, year_start, year_end)
                        VALUES (:id, :name, :make_id, :year_start, :year_end)
                        ON CONFLICT (id) DO UPDATE SET
                            name = EXCLUDED.name,
                            year_start = LEAST(vehicle_models.year_start, EXCLUDED.year_start),
                            year_end = GREATEST(vehicle_models.year_end, EXCLUDED.year_end)
                    """
                    )

                    conn.execute(
                        query,
                        {
                            "id": model.normalized_id,
                            "name": model.model_name,
                            "make_id": make_id,
                            "year_start": model.year or DEFAULT_START_YEAR,
                            "year_end": model.year or DEFAULT_END_YEAR,
                        },
                    )
                    saved_count += 1
                except Exception as e:
                    logger.warning(
                        f"Failed to save model {model.make_name} {model.model_name}: {e}"
                    )

        stats.models_saved += saved_count
        return saved_count

    def save_recalls(self, records: List[RecallRecord], stats: SyncStats) -> int:
        """Save recall records to database."""
        if not records:
            return 0

        engine = self._get_engine()
        saved_count = 0

        with engine.begin() as conn:
            for record in records:
                try:
                    query = text(
                        """
                        INSERT INTO vehicle_recalls (
                            campaign_number, nhtsa_id, manufacturer, make, model,
                            model_year, recall_date, component, summary, consequence,
                            remedy, notes, extracted_dtc_codes, raw_response
                        ) VALUES (
                            :campaign_number, :nhtsa_id, :manufacturer, :make, :model,
                            :model_year, :recall_date, :component, :summary, :consequence,
                            :remedy, :notes, :extracted_dtc_codes, :raw_response
                        )
                        ON CONFLICT (campaign_number) DO NOTHING
                    """
                    )

                    conn.execute(
                        query,
                        {
                            "campaign_number": record.campaign_number,
                            "nhtsa_id": record.nhtsa_id,
                            "manufacturer": record.manufacturer,
                            "make": record.make,
                            "model": record.model,
                            "model_year": record.model_year,
                            "recall_date": record.recall_date,
                            "component": record.component,
                            "summary": record.summary,
                            "consequence": record.consequence,
                            "remedy": record.remedy,
                            "notes": record.notes,
                            "extracted_dtc_codes": record.extracted_dtc_codes,
                            "raw_response": json.dumps(record.raw_response),
                        },
                    )
                    saved_count += 1
                except Exception as e:
                    logger.warning(
                        f"Failed to save recall {record.campaign_number}: {e}"
                    )

        stats.recalls_saved += saved_count
        return saved_count

    def save_complaints(
        self, records: List[ComplaintRecord], stats: SyncStats
    ) -> int:
        """Save complaint records to database."""
        if not records:
            return 0

        engine = self._get_engine()
        saved_count = 0

        with engine.begin() as conn:
            for record in records:
                try:
                    query = text(
                        """
                        INSERT INTO vehicle_complaints (
                            odi_number, manufacturer, make, model, model_year,
                            crash, fire, injuries, deaths, complaint_date,
                            date_of_incident, components, summary, extracted_dtc_codes,
                            raw_response
                        ) VALUES (
                            :odi_number, :manufacturer, :make, :model, :model_year,
                            :crash, :fire, :injuries, :deaths, :complaint_date,
                            :date_of_incident, :components, :summary, :extracted_dtc_codes,
                            :raw_response
                        )
                        ON CONFLICT (odi_number) DO NOTHING
                    """
                    )

                    conn.execute(
                        query,
                        {
                            "odi_number": record.odi_number,
                            "manufacturer": record.manufacturer,
                            "make": record.make,
                            "model": record.model,
                            "model_year": record.model_year,
                            "crash": record.crash,
                            "fire": record.fire,
                            "injuries": record.injuries,
                            "deaths": record.deaths,
                            "complaint_date": record.complaint_date,
                            "date_of_incident": record.date_of_incident,
                            "components": record.components,
                            "summary": record.summary,
                            "extracted_dtc_codes": record.extracted_dtc_codes,
                            "raw_response": json.dumps(record.raw_response),
                        },
                    )
                    saved_count += 1
                except Exception as e:
                    logger.warning(f"Failed to save complaint {record.odi_number}: {e}")

        stats.complaints_saved += saved_count
        return saved_count

    def log_vehicle_sync(
        self,
        make: str,
        model: Optional[str],
        year: Optional[int],
        data_type: str,
        records_count: int,
        status: str = "completed",
        error: Optional[str] = None,
    ) -> None:
        """Log sync operation for tracking."""
        engine = self._get_engine()

        query = text(
            """
            INSERT INTO nhtsa_sync_log (
                make, model, model_year, data_type, records_synced,
                sync_status, error_message
            ) VALUES (
                :make, :model, :model_year, :data_type, :records_synced,
                :sync_status, :error_message
            )
            ON CONFLICT (make, model, model_year, data_type)
            DO UPDATE SET
                records_synced = :records_synced,
                sync_status = :sync_status,
                error_message = :error_message,
                last_synced_at = NOW()
        """
        )

        try:
            with engine.begin() as conn:
                conn.execute(
                    query,
                    {
                        "make": make,
                        "model": model,
                        "model_year": year or 0,
                        "data_type": data_type,
                        "records_synced": records_count,
                        "sync_status": status,
                        "error_message": error,
                    },
                )
        except Exception as e:
            logger.warning(f"Failed to log sync: {e}")

    def get_make_count(self) -> int:
        """Get current count of makes in database."""
        engine = self._get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM vehicle_makes"))
            return result.scalar() or 0

    def get_model_count(self) -> int:
        """Get current count of models in database."""
        engine = self._get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM vehicle_models"))
            return result.scalar() or 0

    def close(self) -> None:
        """Close database connection."""
        if self._engine:
            self._engine.dispose()
            self._engine = None


# =============================================================================
# Sync Orchestrator
# =============================================================================


class NHTSAVehicleSyncOrchestrator:
    """Orchestrates the complete NHTSA vehicle data sync."""

    def __init__(
        self,
        database_url: Optional[str] = None,
        start_year: int = DEFAULT_START_YEAR,
        end_year: int = DEFAULT_END_YEAR,
        limit: Optional[int] = None,
        resume: bool = False,
        dry_run: bool = False,
    ):
        self.start_year = start_year
        self.end_year = end_year
        self.limit = limit
        self.resume = resume
        self.dry_run = dry_run

        self.stats = SyncStats()
        self.rate_limiter = RateLimiter()
        self.client = NHTSAVehicleClient(self.rate_limiter, self.stats)

        # Load or create progress
        self.progress = self._load_progress() if resume else SyncProgress()

        # Database manager
        if database_url and HAS_SQLALCHEMY and not dry_run:
            self.db = DatabaseManager(database_url)
        else:
            self.db = None
            if not dry_run and not database_url:
                logger.warning("No DATABASE_URL provided, running in dry-run mode")
            self.dry_run = True

    def _load_progress(self) -> SyncProgress:
        """Load progress from file."""
        data = load_json(PROGRESS_FILE)
        if data:
            logger.info("Loaded progress from previous run")
            return SyncProgress.from_dict(data)
        return SyncProgress()

    def _save_progress(self) -> None:
        """Save progress to file."""
        self.progress.last_updated = datetime.now(timezone.utc).isoformat()
        save_json(self.progress.to_dict(), PROGRESS_FILE)

    async def sync_makes(self) -> List[VehicleMake]:
        """Sync all vehicle makes."""
        if self.resume and self.progress.makes_synced:
            logger.info(
                f"Skipping makes sync (already synced: {self.progress.makes_count})"
            )
            return []

        logger.info("=" * 60)
        logger.info("SYNCING VEHICLE MAKES")
        logger.info("=" * 60)

        makes = await self.client.get_all_makes()

        if self.limit:
            makes = makes[: self.limit]
            logger.info(f"Limited to {self.limit} makes")

        # Save to file
        makes_data = {
            "metadata": {
                "synced_at": datetime.now(timezone.utc).isoformat(),
                "total_makes": len(makes),
            },
            "makes": [
                {"make_id": m.make_id, "make_name": m.make_name} for m in makes
            ],
        }
        save_json(makes_data, MAKES_FILE)

        # Save to database
        if self.db and not self.dry_run:
            saved = self.db.save_makes(makes, self.stats)
            logger.info(f"Saved {saved} makes to database")

        # Update progress
        self.progress.makes_synced = True
        self.progress.makes_count = len(makes)
        self._save_progress()

        logger.info(f"Synced {len(makes)} vehicle makes")
        return makes

    async def sync_models(
        self, makes: Optional[List[str]] = None
    ) -> Dict[str, List[VehicleModel]]:
        """Sync models for specified makes (or top makes)."""
        target_makes = makes or TOP_MAKES

        logger.info("=" * 60)
        logger.info(f"SYNCING VEHICLE MODELS FOR {len(target_makes)} MAKES")
        logger.info("=" * 60)

        all_models: Dict[str, List[VehicleModel]] = {}

        # Filter out already synced makes if resuming
        if self.resume:
            target_makes = [
                m for m in target_makes if m not in self.progress.models_synced_makes
            ]
            if not target_makes:
                logger.info("All target makes already synced")
                return all_models

        with tqdm(total=len(target_makes), desc="Syncing makes", unit="make") as pbar:
            for make in target_makes:
                try:
                    models = await self.client.get_models_for_make(make)

                    if models:
                        all_models[make] = models
                        logger.info(f"  {make}: {len(models)} models")

                        # Save to database
                        if self.db and not self.dry_run:
                            self.db.save_models(models, self.stats)

                    # Update progress
                    self.progress.models_synced_makes.append(make)
                    self._save_progress()

                except Exception as e:
                    logger.error(f"Failed to sync models for {make}: {e}")
                    self.progress.errors.append(f"models:{make}:{str(e)}")

                pbar.update(1)

        # Save all models to file
        models_data = {
            "metadata": {
                "synced_at": datetime.now(timezone.utc).isoformat(),
                "total_makes": len(all_models),
                "total_models": sum(len(m) for m in all_models.values()),
            },
            "models_by_make": {
                make: [
                    {
                        "model_id": m.model_id,
                        "model_name": m.model_name,
                        "make_id": m.make_id,
                    }
                    for m in models
                ]
                for make, models in all_models.items()
            },
        }
        save_json(models_data, MODELS_FILE)

        total_models = sum(len(m) for m in all_models.values())
        logger.info(f"Synced {total_models} models across {len(all_models)} makes")
        return all_models

    async def sync_recalls(
        self, makes: Optional[List[str]] = None
    ) -> List[RecallRecord]:
        """Sync recalls for specified makes."""
        target_makes = makes or TOP_MAKES

        logger.info("=" * 60)
        logger.info(f"SYNCING RECALLS FOR {len(target_makes)} MAKES")
        logger.info(f"Years: {self.start_year} - {self.end_year}")
        logger.info("=" * 60)

        all_recalls: List[RecallRecord] = []

        total_iterations = len(target_makes) * (self.end_year - self.start_year + 1)

        with tqdm(total=total_iterations, desc="Syncing recalls", unit="req") as pbar:
            for make in target_makes:
                for year in range(self.start_year, self.end_year + 1):
                    vehicle_key = f"{make}:*:{year}"

                    # Check if already synced
                    if (
                        self.resume
                        and vehicle_key in self.progress.recalls_synced_vehicles
                    ):
                        pbar.update(1)
                        continue

                    try:
                        # Get models for this make/year
                        models = await self.client.get_models_for_make_year(make, year)

                        if not models:
                            pbar.update(1)
                            continue

                        # Limit models per make/year
                        if self.limit:
                            models = models[:5]

                        for model in models:
                            recalls = await self.client.get_recalls(
                                make, model.model_name, year
                            )

                            if recalls:
                                all_recalls.extend(recalls)

                                # Save to database
                                if self.db and not self.dry_run:
                                    self.db.save_recalls(recalls, self.stats)
                                    self.db.log_vehicle_sync(
                                        make,
                                        model.model_name,
                                        year,
                                        "recalls",
                                        len(recalls),
                                    )

                        # Update progress
                        self.progress.recalls_synced_vehicles.append(vehicle_key)
                        self._save_progress()

                    except Exception as e:
                        logger.error(f"Failed to sync recalls for {make} {year}: {e}")
                        self.progress.errors.append(f"recalls:{make}:{year}:{str(e)}")

                    pbar.update(1)

        logger.info(f"Synced {len(all_recalls)} recalls")
        return all_recalls

    async def sync_complaints(
        self, makes: Optional[List[str]] = None
    ) -> List[ComplaintRecord]:
        """Sync complaints for specified makes."""
        target_makes = makes or TOP_MAKES

        logger.info("=" * 60)
        logger.info(f"SYNCING COMPLAINTS FOR {len(target_makes)} MAKES")
        logger.info(f"Years: {self.start_year} - {self.end_year}")
        logger.info("=" * 60)

        all_complaints: List[ComplaintRecord] = []

        total_iterations = len(target_makes) * (self.end_year - self.start_year + 1)

        with tqdm(
            total=total_iterations, desc="Syncing complaints", unit="req"
        ) as pbar:
            for make in target_makes:
                for year in range(self.start_year, self.end_year + 1):
                    vehicle_key = f"{make}:*:{year}"

                    # Check if already synced
                    if (
                        self.resume
                        and vehicle_key in self.progress.complaints_synced_vehicles
                    ):
                        pbar.update(1)
                        continue

                    try:
                        # Get models for this make/year
                        models = await self.client.get_models_for_make_year(make, year)

                        if not models:
                            pbar.update(1)
                            continue

                        # Limit models per make/year
                        if self.limit:
                            models = models[:5]

                        for model in models:
                            complaints = await self.client.get_complaints(
                                make, model.model_name, year
                            )

                            if complaints:
                                all_complaints.extend(complaints)

                                # Save to database
                                if self.db and not self.dry_run:
                                    self.db.save_complaints(complaints, self.stats)
                                    self.db.log_vehicle_sync(
                                        make,
                                        model.model_name,
                                        year,
                                        "complaints",
                                        len(complaints),
                                    )

                        # Update progress
                        self.progress.complaints_synced_vehicles.append(vehicle_key)
                        self._save_progress()

                    except Exception as e:
                        logger.error(
                            f"Failed to sync complaints for {make} {year}: {e}"
                        )
                        self.progress.errors.append(
                            f"complaints:{make}:{year}:{str(e)}"
                        )

                    pbar.update(1)

        logger.info(f"Synced {len(all_complaints)} complaints")
        return all_complaints

    async def run(
        self,
        sync_makes: bool = True,
        sync_models: bool = True,
        sync_recalls: bool = False,
        sync_complaints: bool = False,
        target_makes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run sync operations."""
        logger.info("=" * 70)
        logger.info("NHTSA VEHICLE DATA SYNC")
        logger.info("=" * 70)
        logger.info(f"Sync makes: {sync_makes}")
        logger.info(f"Sync models: {sync_models}")
        logger.info(f"Sync recalls: {sync_recalls}")
        logger.info(f"Sync complaints: {sync_complaints}")
        logger.info(f"Years: {self.start_year} - {self.end_year}")
        logger.info(f"Limit: {self.limit or 'None'}")
        logger.info(f"Resume: {self.resume}")
        logger.info(f"Dry run: {self.dry_run}")
        logger.info("=" * 70)

        try:
            # Sync makes
            makes = []
            if sync_makes:
                makes = await self.sync_makes()

            # Sync models
            if sync_models:
                await self.sync_models(target_makes)

            # Sync recalls
            if sync_recalls:
                await self.sync_recalls(target_makes)

            # Sync complaints
            if sync_complaints:
                await self.sync_complaints(target_makes)

        except Exception as e:
            logger.error(f"Sync failed: {e}")
            raise

        finally:
            await self.client.close()
            if self.db:
                self.db.close()

        return self.generate_report()

    def generate_report(self) -> Dict[str, Any]:
        """Generate sync report."""
        db_stats = {}
        if self.db and not self.dry_run:
            try:
                db_stats = {
                    "makes_in_db": self.db.get_make_count(),
                    "models_in_db": self.db.get_model_count(),
                }
            except Exception as e:
                logger.warning(f"Could not get database stats: {e}")

        report = {
            "summary": self.stats.to_dict(),
            "database": db_stats,
            "progress": {
                "makes_synced": self.progress.makes_synced,
                "makes_count": self.progress.makes_count,
                "models_synced_makes": len(self.progress.models_synced_makes),
                "recalls_synced": len(self.progress.recalls_synced_vehicles),
                "complaints_synced": len(self.progress.complaints_synced_vehicles),
            },
            "errors": self.progress.errors[-10:],  # Last 10 errors
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return report


# =============================================================================
# CLI Entry Point
# =============================================================================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Comprehensive NHTSA vehicle data sync for AutoCognitix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Sync all makes (10,000+)
    python scripts/sync_nhtsa_vehicles.py --makes-only

    # Sync models for top manufacturers
    python scripts/sync_nhtsa_vehicles.py --models-only

    # Sync recalls for specific makes
    python scripts/sync_nhtsa_vehicles.py --recalls --makes Toyota Honda

    # Full sync with all options
    python scripts/sync_nhtsa_vehicles.py --makes-only --models-only --recalls --complaints

    # Resume interrupted sync
    python scripts/sync_nhtsa_vehicles.py --resume

    # Dry run (no database writes)
    python scripts/sync_nhtsa_vehicles.py --dry-run --makes-only
        """,
    )

    parser.add_argument(
        "--makes-only",
        action="store_true",
        help="Only sync vehicle makes (~10,000+)",
    )
    parser.add_argument(
        "--models-only",
        action="store_true",
        help="Only sync vehicle models for top manufacturers",
    )
    parser.add_argument(
        "--recalls",
        action="store_true",
        help="Sync recall data for popular vehicles",
    )
    parser.add_argument(
        "--complaints",
        action="store_true",
        help="Sync complaint data for popular vehicles",
    )
    parser.add_argument(
        "--makes",
        type=str,
        nargs="+",
        help="Specific make(s) to target for models/recalls/complaints",
    )
    parser.add_argument(
        "--years",
        type=str,
        default=f"{DEFAULT_START_YEAR}-{DEFAULT_END_YEAR}",
        help=f"Year range for recalls/complaints (default: {DEFAULT_START_YEAR}-{DEFAULT_END_YEAR})",
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        help="Limit number of items to process",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous progress",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch data but don't save to database",
    )
    parser.add_argument(
        "--database-url",
        type=str,
        default=os.environ.get("DATABASE_URL"),
        help="PostgreSQL database URL (default: from DATABASE_URL env)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save report to JSON file",
    )

    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()

    # Reconfigure logging
    global logger
    logger = setup_logging(verbose=args.verbose, log_file=LOG_FILE)

    # Parse year range
    try:
        year_parts = args.years.split("-")
        start_year = int(year_parts[0])
        end_year = int(year_parts[1]) if len(year_parts) > 1 else start_year
    except (ValueError, IndexError):
        logger.error(f"Invalid year range: {args.years}")
        sys.exit(1)

    # Determine what to sync
    sync_makes = args.makes_only
    sync_models = args.models_only
    sync_recalls = args.recalls
    sync_complaints = args.complaints

    # If nothing specified, default to makes only
    if not any([sync_makes, sync_models, sync_recalls, sync_complaints]):
        sync_makes = True
        logger.info("No sync type specified, defaulting to --makes-only")

    # Check database URL
    if not args.dry_run and not args.database_url:
        logger.warning("No DATABASE_URL provided, running in dry-run mode")
        args.dry_run = True

    if not args.dry_run and not HAS_SQLALCHEMY:
        logger.error(
            "SQLAlchemy not installed. Install with: pip install sqlalchemy psycopg2-binary"
        )
        sys.exit(1)

    # Create orchestrator
    orchestrator = NHTSAVehicleSyncOrchestrator(
        database_url=args.database_url,
        start_year=start_year,
        end_year=end_year,
        limit=args.limit,
        resume=args.resume,
        dry_run=args.dry_run,
    )

    # Run sync
    report = await orchestrator.run(
        sync_makes=sync_makes,
        sync_models=sync_models,
        sync_recalls=sync_recalls,
        sync_complaints=sync_complaints,
        target_makes=args.makes,
    )

    # Print report
    print("\n" + "=" * 70)
    print("SYNC REPORT")
    print("=" * 70)
    print(f"Makes fetched: {report['summary']['makes']['fetched']}")
    print(f"Makes saved: {report['summary']['makes']['saved']}")
    print(f"Models fetched: {report['summary']['models']['fetched']}")
    print(f"Models saved: {report['summary']['models']['saved']}")
    print(f"Recalls fetched: {report['summary']['recalls']['fetched']}")
    print(f"Recalls saved: {report['summary']['recalls']['saved']}")
    print(f"Complaints fetched: {report['summary']['complaints']['fetched']}")
    print(f"Complaints saved: {report['summary']['complaints']['saved']}")
    print(f"DTC codes extracted: {report['summary']['dtc_codes_extracted']}")
    print(f"API requests: {report['summary']['api']['requests']}")
    print(f"API errors: {report['summary']['api']['errors']}")
    print(f"Elapsed time: {report['summary']['elapsed_seconds']}s")

    if report.get("database"):
        print(f"\nDatabase:")
        print(f"  Makes in DB: {report['database'].get('makes_in_db', 'N/A')}")
        print(f"  Models in DB: {report['database'].get('models_in_db', 'N/A')}")

    if report.get("errors"):
        print(f"\nRecent errors:")
        for error in report["errors"][-5:]:
            print(f"  - {error}")

    print("=" * 70)

    # Save report to file
    if args.output:
        output_path = Path(args.output)
        save_json(report, output_path)
        print(f"Report saved to: {output_path}")

    return report


if __name__ == "__main__":
    asyncio.run(main())
