#!/usr/bin/env python3
"""
Complete NHTSA Data Sync Script for AutoCognitix.

Comprehensive sync of vehicle recalls and complaints from the NHTSA API
for ALL major vehicle makes with PostgreSQL storage.

Features:
- Parallel requests per make (10 req/second max)
- Incremental sync (skip already synced data)
- PostgreSQL storage (vehicle_recalls, vehicle_complaints tables)
- DTC code extraction and correlation
- Compressed data storage (JSONB)
- Detailed progress reporting

NHTSA API Documentation:
- Recalls: https://api.nhtsa.gov/recalls/recallsByVehicle
- Complaints: https://api.nhtsa.gov/complaints/complaintsByVehicle
- VPIC (VIN/Models): https://vpic.nhtsa.dot.gov/api/

Usage:
    python scripts/sync_nhtsa_complete.py                    # Full sync
    python scripts/sync_nhtsa_complete.py --incremental      # Skip already synced
    python scripts/sync_nhtsa_complete.py --make Toyota      # Single make
    python scripts/sync_nhtsa_complete.py --years 2020-2025  # Year range
    python scripts/sync_nhtsa_complete.py --dry-run          # Preview only
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Try to import database dependencies
try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("nhtsa_sync")


# =============================================================================
# Configuration
# =============================================================================

# NHTSA API endpoints
NHTSA_BASE_URL = "https://api.nhtsa.gov"
VPIC_BASE_URL = "https://vpic.nhtsa.dot.gov/api/vehicles"

RECALLS_ENDPOINT = f"{NHTSA_BASE_URL}/recalls/recallsByVehicle"
COMPLAINTS_ENDPOINT = f"{NHTSA_BASE_URL}/complaints/complaintsByVehicle"

# Rate limiting: 10 requests per second max
MAX_REQUESTS_PER_SECOND = 10
REQUEST_INTERVAL = 1.0 / MAX_REQUESTS_PER_SECOND  # 0.1 seconds

# Default year range
DEFAULT_START_YEAR = 2010
DEFAULT_END_YEAR = 2025

# ALL major vehicle makes (30 makes as specified)
ALL_MAKES = [
    "Toyota",
    "Honda",
    "Ford",
    "Chevrolet",
    "Nissan",
    "Volkswagen",
    "BMW",
    "Mercedes-Benz",
    "Audi",
    "Hyundai",
    "Kia",
    "Mazda",
    "Subaru",
    "Jeep",
    "Ram",
    "GMC",
    "Dodge",
    "Chrysler",
    "Lexus",
    "Acura",
    "Infiniti",
    "Volvo",
    "Porsche",
    "Land Rover",
    "Jaguar",
    "Mini",
    "Fiat",
    "Alfa Romeo",
    "Mitsubishi",
    "Suzuki",
]

# DTC code patterns
DTC_PATTERN = re.compile(r'\b([PCBU][0-9A-Fa-f]{4})\b', re.IGNORECASE)
EXTENDED_DTC_PATTERNS = [
    re.compile(r'DTC\s*[:\-]?\s*([PCBU][0-9A-Fa-f]{4})', re.IGNORECASE),
    re.compile(r'code\s*[:\-]?\s*([PCBU][0-9A-Fa-f]{4})', re.IGNORECASE),
    re.compile(r'trouble\s+code\s*[:\-]?\s*([PCBU][0-9A-Fa-f]{4})', re.IGNORECASE),
    re.compile(r'error\s+code\s*[:\-]?\s*([PCBU][0-9A-Fa-f]{4})', re.IGNORECASE),
]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SyncStats:
    """Statistics for sync operation."""
    recalls_fetched: int = 0
    recalls_new: int = 0
    recalls_updated: int = 0
    recalls_skipped: int = 0
    complaints_fetched: int = 0
    complaints_new: int = 0
    complaints_updated: int = 0
    complaints_skipped: int = 0
    dtc_correlations_found: int = 0
    api_requests: int = 0
    api_errors: int = 0
    start_time: float = field(default_factory=time.time)

    def elapsed_time(self) -> float:
        return time.time() - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recalls": {
                "fetched": self.recalls_fetched,
                "new": self.recalls_new,
                "updated": self.recalls_updated,
                "skipped": self.recalls_skipped,
            },
            "complaints": {
                "fetched": self.complaints_fetched,
                "new": self.complaints_new,
                "updated": self.complaints_updated,
                "skipped": self.complaints_skipped,
            },
            "dtc_correlations_found": self.dtc_correlations_found,
            "api_requests": self.api_requests,
            "api_errors": self.api_errors,
            "elapsed_seconds": self.elapsed_time(),
        }


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
# DTC Code Extraction
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

    # NHTSA uses various date formats
    formats = [
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%Y/%m/%d",
        "%d/%m/%Y",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str[:10], fmt).date()
        except (ValueError, TypeError):
            continue

    return None


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

class NHTSAClient:
    """Async NHTSA API client with rate limiting and error handling."""

    def __init__(self, rate_limiter: RateLimiter, stats: SyncStats):
        self.rate_limiter = rate_limiter
        self.stats = stats
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
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

    async def _request(self, url: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make rate-limited API request with error handling."""
        await self.rate_limiter.acquire()
        self.stats.api_requests += 1

        client = await self._get_client()

        try:
            response = await client.get(url, params=params)

            if response.status_code == 429:
                # Rate limited - wait and retry
                retry_after = int(response.headers.get("Retry-After", 60))
                logger.warning(f"Rate limited, waiting {retry_after}s...")
                await asyncio.sleep(retry_after)
                return await self._request(url, params)

            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP {e.response.status_code}: {url}")
            self.stats.api_errors += 1
            return None
        except httpx.TimeoutException:
            logger.error(f"Timeout: {url}")
            self.stats.api_errors += 1
            return None
        except Exception as e:
            logger.error(f"Request error: {e}")
            self.stats.api_errors += 1
            return None

    async def get_models_for_make(self, make: str, year: int) -> List[str]:
        """Get available models for a make and year."""
        url = f"{VPIC_BASE_URL}/GetModelsForMakeYear/make/{make}/modelyear/{year}"
        data = await self._request(url, {"format": "json"})

        if not data:
            return []

        results = data.get("Results", [])
        return [r.get("Model_Name", "") for r in results if r.get("Model_Name")]

    async def get_recalls(
        self, make: str, model: str, year: int
    ) -> List[RecallRecord]:
        """Fetch recalls for a specific vehicle."""
        data = await self._request(RECALLS_ENDPOINT, {
            "make": make,
            "model": model,
            "modelYear": year,
        })

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
                self.stats.dtc_correlations_found += len(dtc_codes)

        return records

    async def get_complaints(
        self, make: str, model: str, year: int
    ) -> List[ComplaintRecord]:
        """Fetch complaints for a specific vehicle."""
        data = await self._request(COMPLAINTS_ENDPOINT, {
            "make": make,
            "model": model,
            "modelYear": year,
        })

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
                self.stats.dtc_correlations_found += len(dtc_codes)

        return records

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


# =============================================================================
# Database Operations
# =============================================================================

class DatabaseManager:
    """Manages PostgreSQL database operations."""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self._engine = None

    def _get_sync_engine(self):
        """Get synchronous engine for database operations."""
        if self._engine is None:
            # Convert async URL to sync for simple operations
            sync_url = self.database_url.replace("+asyncpg", "")
            self._engine = create_engine(sync_url, pool_pre_ping=True)
        return self._engine

    def get_synced_records(self, make: str, data_type: str) -> Set[Tuple[str, int]]:
        """Get already synced (model, year) combinations for incremental sync."""
        engine = self._get_sync_engine()

        query = text("""
            SELECT DISTINCT model, model_year
            FROM nhtsa_sync_log
            WHERE make = :make
            AND data_type = :data_type
            AND sync_status = 'completed'
        """)

        with engine.connect() as conn:
            result = conn.execute(query, {"make": make, "data_type": data_type})
            return {(row[0], row[1]) for row in result}

    def get_existing_campaign_numbers(self) -> Set[str]:
        """Get existing recall campaign numbers."""
        engine = self._get_sync_engine()

        with engine.connect() as conn:
            result = conn.execute(text("SELECT campaign_number FROM vehicle_recalls"))
            return {row[0] for row in result}

    def get_existing_odi_numbers(self) -> Set[str]:
        """Get existing complaint ODI numbers."""
        engine = self._get_sync_engine()

        with engine.connect() as conn:
            result = conn.execute(text("SELECT odi_number FROM vehicle_complaints"))
            return {row[0] for row in result}

    def save_recalls(self, records: List[RecallRecord], stats: SyncStats) -> int:
        """Save recall records to database."""
        if not records:
            return 0

        engine = self._get_sync_engine()
        existing = self.get_existing_campaign_numbers()

        new_count = 0
        with engine.begin() as conn:
            for record in records:
                if record.campaign_number in existing:
                    stats.recalls_skipped += 1
                    continue

                query = text("""
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
                """)

                conn.execute(query, {
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
                })

                existing.add(record.campaign_number)
                new_count += 1
                stats.recalls_new += 1

        return new_count

    def save_complaints(self, records: List[ComplaintRecord], stats: SyncStats) -> int:
        """Save complaint records to database."""
        if not records:
            return 0

        engine = self._get_sync_engine()
        existing = self.get_existing_odi_numbers()

        new_count = 0
        with engine.begin() as conn:
            for record in records:
                if record.odi_number in existing:
                    stats.complaints_skipped += 1
                    continue

                query = text("""
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
                """)

                conn.execute(query, {
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
                })

                existing.add(record.odi_number)
                new_count += 1
                stats.complaints_new += 1

        return new_count

    def log_sync(
        self,
        make: str,
        model: Optional[str],
        year: int,
        data_type: str,
        records_count: int,
        status: str = "completed",
        error: Optional[str] = None,
    ) -> None:
        """Log sync operation for incremental tracking."""
        engine = self._get_sync_engine()

        query = text("""
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
        """)

        with engine.begin() as conn:
            conn.execute(query, {
                "make": make,
                "model": model,
                "model_year": year,
                "data_type": data_type,
                "records_synced": records_count,
                "sync_status": status,
                "error_message": error,
            })

    def create_dtc_correlations(self, stats: SyncStats) -> int:
        """Create DTC code correlations from extracted codes."""
        engine = self._get_sync_engine()
        correlations_created = 0

        with engine.begin() as conn:
            # Create correlations for recalls
            recall_query = text("""
                INSERT INTO dtc_recall_correlations (dtc_code, recall_id, confidence, extraction_method)
                SELECT DISTINCT
                    unnest(vr.extracted_dtc_codes) as dtc_code,
                    vr.id as recall_id,
                    1.0 as confidence,
                    'explicit' as extraction_method
                FROM vehicle_recalls vr
                WHERE vr.extracted_dtc_codes IS NOT NULL
                AND array_length(vr.extracted_dtc_codes, 1) > 0
                AND EXISTS (SELECT 1 FROM dtc_codes dc WHERE dc.code = unnest(vr.extracted_dtc_codes))
                ON CONFLICT (dtc_code, recall_id) DO NOTHING
            """)

            result = conn.execute(recall_query)
            correlations_created += result.rowcount

            # Create correlations for complaints
            complaint_query = text("""
                INSERT INTO dtc_complaint_correlations (dtc_code, complaint_id, confidence, extraction_method)
                SELECT DISTINCT
                    unnest(vc.extracted_dtc_codes) as dtc_code,
                    vc.id as complaint_id,
                    1.0 as confidence,
                    'explicit' as extraction_method
                FROM vehicle_complaints vc
                WHERE vc.extracted_dtc_codes IS NOT NULL
                AND array_length(vc.extracted_dtc_codes, 1) > 0
                AND EXISTS (SELECT 1 FROM dtc_codes dc WHERE dc.code = unnest(vc.extracted_dtc_codes))
                ON CONFLICT (dtc_code, complaint_id) DO NOTHING
            """)

            result = conn.execute(complaint_query)
            correlations_created += result.rowcount

        return correlations_created

    def close(self) -> None:
        if self._engine:
            self._engine.dispose()
            self._engine = None


# =============================================================================
# Sync Orchestrator
# =============================================================================

class NHTSASyncOrchestrator:
    """Orchestrates the complete NHTSA data sync."""

    def __init__(
        self,
        database_url: Optional[str] = None,
        makes: Optional[List[str]] = None,
        start_year: int = DEFAULT_START_YEAR,
        end_year: int = DEFAULT_END_YEAR,
        incremental: bool = False,
        dry_run: bool = False,
    ):
        self.makes = makes or ALL_MAKES
        self.start_year = start_year
        self.end_year = end_year
        self.incremental = incremental
        self.dry_run = dry_run

        self.stats = SyncStats()
        self.rate_limiter = RateLimiter()
        self.client = NHTSAClient(self.rate_limiter, self.stats)

        if database_url and HAS_SQLALCHEMY and not dry_run:
            self.db = DatabaseManager(database_url)
        else:
            self.db = None

    async def sync_make(self, make: str) -> Tuple[List[RecallRecord], List[ComplaintRecord]]:
        """Sync all data for a single make."""
        logger.info(f"Syncing {make}...")

        all_recalls = []
        all_complaints = []

        # Get synced records for incremental mode
        synced_recalls = set()
        synced_complaints = set()

        if self.incremental and self.db:
            synced_recalls = self.db.get_synced_records(make, "recalls")
            synced_complaints = self.db.get_synced_records(make, "complaints")

        for year in range(self.start_year, self.end_year + 1):
            # Get models for this make/year
            models = await self.client.get_models_for_make(make, year)

            if not models:
                logger.debug(f"  No models for {make} {year}")
                continue

            for model in models:
                # Skip if already synced (incremental mode)
                skip_recalls = (model, year) in synced_recalls
                skip_complaints = (model, year) in synced_complaints

                if skip_recalls and skip_complaints:
                    logger.debug(f"  Skipping {make} {model} {year} (already synced)")
                    continue

                # Fetch recalls
                if not skip_recalls:
                    recalls = await self.client.get_recalls(make, model, year)
                    if recalls:
                        all_recalls.extend(recalls)
                        logger.info(f"  {make} {model} {year}: {len(recalls)} recalls")

                        if self.db and not self.dry_run:
                            self.db.save_recalls(recalls, self.stats)
                            self.db.log_sync(make, model, year, "recalls", len(recalls))

                # Fetch complaints
                if not skip_complaints:
                    complaints = await self.client.get_complaints(make, model, year)
                    if complaints:
                        all_complaints.extend(complaints)
                        logger.info(f"  {make} {model} {year}: {len(complaints)} complaints")

                        if self.db and not self.dry_run:
                            self.db.save_complaints(complaints, self.stats)
                            self.db.log_sync(make, model, year, "complaints", len(complaints))

        return all_recalls, all_complaints

    async def sync_all(self) -> Dict[str, Any]:
        """Run complete sync for all makes."""
        logger.info("=" * 70)
        logger.info("NHTSA COMPLETE DATA SYNC")
        logger.info("=" * 70)
        logger.info(f"Makes: {len(self.makes)}")
        logger.info(f"Years: {self.start_year}-{self.end_year}")
        logger.info(f"Incremental: {self.incremental}")
        logger.info(f"Dry run: {self.dry_run}")
        logger.info("=" * 70)

        all_recalls = []
        all_complaints = []

        try:
            # Process makes in parallel batches
            # Each make is processed sequentially to respect rate limits
            for make in self.makes:
                recalls, complaints = await self.sync_make(make)
                all_recalls.extend(recalls)
                all_complaints.extend(complaints)

            # Create DTC correlations
            if self.db and not self.dry_run:
                logger.info("Creating DTC correlations...")
                correlations = self.db.create_dtc_correlations(self.stats)
                logger.info(f"Created {correlations} DTC correlations")

        except Exception as e:
            logger.error(f"Sync failed: {e}")
            raise

        finally:
            await self.client.close()
            if self.db:
                self.db.close()

        return self.generate_report(all_recalls, all_complaints)

    def generate_report(
        self, recalls: List[RecallRecord], complaints: List[ComplaintRecord]
    ) -> Dict[str, Any]:
        """Generate sync report."""
        # Collect unique DTC codes
        all_dtc_codes: Set[str] = set()
        for r in recalls:
            all_dtc_codes.update(r.extracted_dtc_codes)
        for c in complaints:
            all_dtc_codes.update(c.extracted_dtc_codes)

        # Count by make
        recalls_by_make = {}
        complaints_by_make = {}

        for r in recalls:
            recalls_by_make[r.make] = recalls_by_make.get(r.make, 0) + 1

        for c in complaints:
            complaints_by_make[c.make] = complaints_by_make.get(c.make, 0) + 1

        report = {
            "summary": {
                "total_recalls_synced": self.stats.recalls_fetched,
                "total_complaints_synced": self.stats.complaints_fetched,
                "new_recalls": self.stats.recalls_new,
                "new_complaints": self.stats.complaints_new,
                "unique_dtc_codes_found": len(all_dtc_codes),
                "dtc_correlations": self.stats.dtc_correlations_found,
            },
            "api_stats": {
                "requests_made": self.stats.api_requests,
                "errors": self.stats.api_errors,
                "elapsed_seconds": round(self.stats.elapsed_time(), 2),
            },
            "by_make": {
                "recalls": recalls_by_make,
                "complaints": complaints_by_make,
            },
            "dtc_codes_extracted": sorted(all_dtc_codes),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return report


# =============================================================================
# CLI Entry Point
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Complete NHTSA data sync for AutoCognitix"
    )
    parser.add_argument(
        "--make",
        type=str,
        nargs="+",
        help="Specific make(s) to sync (default: all 30 makes)",
    )
    parser.add_argument(
        "--years",
        type=str,
        default=f"{DEFAULT_START_YEAR}-{DEFAULT_END_YEAR}",
        help=f"Year range (default: {DEFAULT_START_YEAR}-{DEFAULT_END_YEAR})",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Skip already synced data",
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
        help="PostgreSQL database URL",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save report to JSON file",
    )

    return parser.parse_args()


async def main():
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
        sys.exit(1)

    # Validate database URL
    if not args.dry_run and not args.database_url:
        logger.warning("No DATABASE_URL provided, running in dry-run mode")
        args.dry_run = True

    if not args.dry_run and not HAS_SQLALCHEMY:
        logger.error("SQLAlchemy not installed. Install with: pip install sqlalchemy psycopg2-binary")
        sys.exit(1)

    # Create orchestrator
    orchestrator = NHTSASyncOrchestrator(
        database_url=args.database_url,
        makes=args.make,
        start_year=start_year,
        end_year=end_year,
        incremental=args.incremental,
        dry_run=args.dry_run,
    )

    # Run sync
    report = await orchestrator.sync_all()

    # Print report
    print("\n" + "=" * 70)
    print("SYNC REPORT")
    print("=" * 70)
    print(f"Total recalls synced: {report['summary']['total_recalls_synced']}")
    print(f"Total complaints synced: {report['summary']['total_complaints_synced']}")
    print(f"New recalls: {report['summary']['new_recalls']}")
    print(f"New complaints: {report['summary']['new_complaints']}")
    print(f"Unique DTC codes found: {report['summary']['unique_dtc_codes_found']}")
    print(f"DTC correlations: {report['summary']['dtc_correlations']}")
    print(f"API requests: {report['api_stats']['requests_made']}")
    print(f"API errors: {report['api_stats']['errors']}")
    print(f"Elapsed time: {report['api_stats']['elapsed_seconds']}s")

    if report['dtc_codes_extracted']:
        print(f"\nTop DTC codes found:")
        for code in report['dtc_codes_extracted'][:20]:
            print(f"  {code}")

    print("=" * 70)

    # Save report to file
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Report saved to: {output_path}")

    return report


if __name__ == "__main__":
    asyncio.run(main())
