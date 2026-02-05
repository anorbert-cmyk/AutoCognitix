#!/usr/bin/env python3
"""
NHTSA Data Sync Script for AutoCognitix.

Fetches vehicle recalls, complaints, and VIN decoding capabilities from the
NHTSA (National Highway Traffic Safety Administration) public API.

Features:
- Fetches recalls for popular European and American makes
- Extracts DTC codes from recall/complaint descriptions
- Saves data to data/nhtsa/recalls.json and data/nhtsa/complaints.json
- Rate limiting (1 second delay between requests)
- Merges extracted DTC codes with master DTC file

NHTSA API Documentation:
- Base URL: https://api.nhtsa.gov/
- No authentication required
- Free tier with rate limits

Usage:
    python scripts/sync_nhtsa.py                    # Fetch all data
    python scripts/sync_nhtsa.py --recalls          # Fetch recalls only
    python scripts/sync_nhtsa.py --complaints       # Fetch complaints only
    python scripts/sync_nhtsa.py --years 2020-2024  # Specify year range
    python scripts/sync_nhtsa.py --merge-dtc        # Merge extracted DTC codes
    python scripts/sync_nhtsa.py --verbose          # Enable verbose logging
"""

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx

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
COMPLAINTS_ENDPOINT = f"{NHTSA_BASE_URL}/complaints/complaintsByVehicle"

# Output paths
DATA_DIR = PROJECT_ROOT / "data" / "nhtsa"
RECALLS_FILE = DATA_DIR / "recalls.json"
COMPLAINTS_FILE = DATA_DIR / "complaints.json"
EXTRACTED_DTC_FILE = DATA_DIR / "extracted_dtc_codes.json"

# Master DTC file for merging
MASTER_DTC_FILE = PROJECT_ROOT / "data" / "dtc_codes" / "all_codes_merged.json"

# Rate limiting
REQUEST_DELAY = 1.0  # seconds between requests

# Popular vehicle makes to fetch (European + American + Japanese)
POPULAR_MAKES = [
    "Volkswagen",
    "BMW",
    "Audi",
    "Mercedes-Benz",
    "Toyota",
    "Honda",
    "Ford",
]

# Default year range
DEFAULT_START_YEAR = 2015
DEFAULT_END_YEAR = 2024

# DTC code pattern (P0xxx, C0xxx, B0xxx, U0xxx)
DTC_PATTERN = re.compile(
    r'\b([PCBU][0-9A-Fa-f]{4})\b',
    re.IGNORECASE
)

# Extended DTC patterns that might appear in text
EXTENDED_DTC_PATTERNS = [
    re.compile(r'DTC\s*[:\-]?\s*([PCBU][0-9A-Fa-f]{4})', re.IGNORECASE),
    re.compile(r'code\s*[:\-]?\s*([PCBU][0-9A-Fa-f]{4})', re.IGNORECASE),
    re.compile(r'trouble\s+code\s*[:\-]?\s*([PCBU][0-9A-Fa-f]{4})', re.IGNORECASE),
    re.compile(r'error\s+code\s*[:\-]?\s*([PCBU][0-9A-Fa-f]{4})', re.IGNORECASE),
]


# =============================================================================
# DTC Code Extraction
# =============================================================================


def extract_dtc_codes(text: str) -> Set[str]:
    """
    Extract DTC codes from text content.

    Searches for standard OBD-II DTC code patterns in recall/complaint
    summaries, consequences, and remedies.

    Args:
        text: Text content to search for DTC codes.

    Returns:
        Set of unique DTC codes found (uppercase).
    """
    if not text:
        return set()

    codes = set()

    # Primary pattern - standard DTC format
    for match in DTC_PATTERN.finditer(text):
        codes.add(match.group(1).upper())

    # Extended patterns
    for pattern in EXTENDED_DTC_PATTERNS:
        for match in pattern.finditer(text):
            codes.add(match.group(1).upper())

    return codes


def get_category_from_code(code: str) -> str:
    """Determine the category from a DTC code prefix."""
    if not code:
        return "unknown"
    prefix = code[0].upper()
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

    # Network codes are typically high severity
    if prefix == "U":
        return "high"

    # Body codes for safety systems are critical
    if prefix == "B" and code.upper().startswith(("B0", "B1")):
        return "critical"

    # Powertrain misfire and transmission codes
    if prefix == "P":
        if code.upper().startswith("P03"):  # Misfire
            return "high"
        if code.upper().startswith(("P07", "P08", "P09")):  # Transmission
            return "high"

    return "medium"


def get_system_from_code(code: str) -> str:
    """Determine the system from a DTC code."""
    if not code or len(code) < 3:
        return ""

    prefix = code[0].upper()
    middle = code[1:3].upper()

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


# =============================================================================
# NHTSA API Client
# =============================================================================


class NHTSASyncClient:
    """
    Synchronous NHTSA API client for batch data fetching.

    Uses httpx with rate limiting to respect API limits.
    """

    def __init__(self, delay: float = REQUEST_DELAY):
        """
        Initialize the NHTSA sync client.

        Args:
            delay: Seconds to wait between requests.
        """
        self.delay = delay
        self._last_request_time = 0.0
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
                headers={
                    "User-Agent": "AutoCognitix/1.0 (Vehicle Diagnostic Platform)",
                    "Accept": "application/json",
                },
            )
        return self._client

    async def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.delay:
            await asyncio.sleep(self.delay - elapsed)
        self._last_request_time = time.time()

    async def _make_request(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Make an API request with rate limiting.

        Args:
            url: Request URL.
            params: Query parameters.

        Returns:
            JSON response or None on error.
        """
        await self._rate_limit()
        client = await self._get_client()

        try:
            logger.debug(f"Requesting: {url} with params: {params}")
            response = await client.get(url, params=params)

            if response.status_code == 429:
                # Rate limited - wait and retry once
                logger.warning("Rate limited by NHTSA API, waiting 60 seconds...")
                await asyncio.sleep(60)
                response = await client.get(url, params=params)

            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code}: {url}")
            return None
        except httpx.TimeoutException:
            logger.error(f"Timeout: {url}")
            return None
        except Exception as e:
            logger.error(f"Request error: {e}")
            return None

    async def get_recalls(
        self,
        make: str,
        model: str,
        year: int,
    ) -> List[Dict[str, Any]]:
        """
        Fetch recalls for a specific vehicle.

        Args:
            make: Vehicle make (e.g., "Toyota").
            model: Vehicle model (e.g., "Camry").
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
            # Extract text fields for DTC code searching
            summary = item.get("Summary", "") or ""
            consequence = item.get("Consequence", "") or ""
            remedy = item.get("Remedy", "") or ""
            notes = item.get("Notes", "") or ""

            # Combine all text for DTC extraction
            full_text = f"{summary} {consequence} {remedy} {notes}"
            dtc_codes = extract_dtc_codes(full_text)

            recall = {
                "campaign_number": item.get("NHTSACampaignNumber", ""),
                "manufacturer": item.get("Manufacturer", make),
                "make": make,
                "model": model,
                "model_year": year,
                "recall_date": item.get("ReportReceivedDate"),
                "component": item.get("Component", "Unknown"),
                "summary": summary,
                "consequence": consequence,
                "remedy": remedy,
                "notes": notes,
                "nhtsa_id": item.get("NHTSAActionNumber"),
                "extracted_dtc_codes": list(dtc_codes),
            }
            recalls.append(recall)

        return recalls

    async def get_complaints(
        self,
        make: str,
        model: str,
        year: int,
    ) -> List[Dict[str, Any]]:
        """
        Fetch complaints for a specific vehicle.

        Args:
            make: Vehicle make (e.g., "Toyota").
            model: Vehicle model (e.g., "Camry").
            year: Model year.

        Returns:
            List of complaint records.
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

        # Transform to standard format
        complaints = []
        for item in results:
            summary = item.get("summary", "") or ""
            components = item.get("components", "") or ""

            # Extract DTC codes from summary
            dtc_codes = extract_dtc_codes(f"{summary} {components}")

            complaint = {
                "odi_number": item.get("odiNumber"),
                "manufacturer": item.get("manufacturer", make),
                "make": make,
                "model": model,
                "model_year": year,
                "crash": item.get("crash", "N") == "Y",
                "fire": item.get("fire", "N") == "Y",
                "injuries": int(item.get("numberOfInjuries") or 0),
                "deaths": int(item.get("numberOfDeaths") or 0),
                "complaint_date": item.get("dateComplaintFiled"),
                "date_of_incident": item.get("dateOfIncident"),
                "components": components,
                "summary": summary,
                "extracted_dtc_codes": list(dtc_codes),
            }
            complaints.append(complaint)

        return complaints

    async def get_models_for_make(self, make: str, year: int) -> List[str]:
        """
        Get available models for a make and year.

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
            return []

        results = data.get("Results", [])
        return [r.get("Model_Name", "") for r in results if r.get("Model_Name")]

    async def decode_vin(self, vin: str) -> Optional[Dict[str, Any]]:
        """
        Decode a VIN using NHTSA VPIC API.

        Args:
            vin: 17-character VIN.

        Returns:
            Decoded vehicle information or None on error.
        """
        if len(vin) != 17:
            logger.error(f"Invalid VIN length: {len(vin)}")
            return None

        url = f"{VPIC_BASE_URL}/DecodeVinValues/{vin}"
        params = {"format": "json"}

        data = await self._make_request(url, params)
        if not data:
            return None

        results = data.get("Results", [{}])[0]

        return {
            "vin": vin,
            "make": results.get("Make"),
            "model": results.get("Model"),
            "model_year": results.get("ModelYear"),
            "body_class": results.get("BodyClass"),
            "vehicle_type": results.get("VehicleType"),
            "manufacturer": results.get("Manufacturer"),
            "plant_country": results.get("PlantCountry"),
            "engine_cylinders": results.get("EngineCylinders"),
            "engine_displacement_l": results.get("DisplacementL"),
            "fuel_type": results.get("FuelTypePrimary"),
            "drive_type": results.get("DriveType"),
            "error_code": results.get("ErrorCode"),
        }

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


# =============================================================================
# Data Processing
# =============================================================================


def save_json(data: Any, file_path: Path) -> None:
    """Save data to JSON file with proper formatting."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved data to {file_path}")


def load_json(file_path: Path) -> Optional[Any]:
    """Load data from JSON file."""
    if not file_path.exists():
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_all_dtc_codes(recalls: List[Dict], complaints: List[Dict]) -> Dict[str, Dict]:
    """
    Extract and consolidate all DTC codes from recalls and complaints.

    Args:
        recalls: List of recall records.
        complaints: List of complaint records.

    Returns:
        Dictionary of DTC codes with metadata.
    """
    dtc_codes: Dict[str, Dict] = {}

    # Process recalls
    for recall in recalls:
        for code in recall.get("extracted_dtc_codes", []):
            if code not in dtc_codes:
                dtc_codes[code] = {
                    "code": code,
                    "recall_references": [],
                    "complaint_references": [],
                    "related_components": set(),
                    "related_makes": set(),
                    "related_models": set(),
                }

            dtc_codes[code]["recall_references"].append({
                "campaign_number": recall.get("campaign_number"),
                "make": recall.get("make"),
                "model": recall.get("model"),
                "year": recall.get("model_year"),
                "component": recall.get("component"),
            })

            dtc_codes[code]["related_components"].add(recall.get("component", ""))
            dtc_codes[code]["related_makes"].add(recall.get("make", ""))
            dtc_codes[code]["related_models"].add(recall.get("model", ""))

    # Process complaints
    for complaint in complaints:
        for code in complaint.get("extracted_dtc_codes", []):
            if code not in dtc_codes:
                dtc_codes[code] = {
                    "code": code,
                    "recall_references": [],
                    "complaint_references": [],
                    "related_components": set(),
                    "related_makes": set(),
                    "related_models": set(),
                }

            dtc_codes[code]["complaint_references"].append({
                "odi_number": complaint.get("odi_number"),
                "make": complaint.get("make"),
                "model": complaint.get("model"),
                "year": complaint.get("model_year"),
                "components": complaint.get("components"),
            })

            dtc_codes[code]["related_components"].add(complaint.get("components", ""))
            dtc_codes[code]["related_makes"].add(complaint.get("make", ""))
            dtc_codes[code]["related_models"].add(complaint.get("model", ""))

    # Convert sets to lists for JSON serialization
    for code_data in dtc_codes.values():
        code_data["related_components"] = [c for c in code_data["related_components"] if c]
        code_data["related_makes"] = [m for m in code_data["related_makes"] if m]
        code_data["related_models"] = [m for m in code_data["related_models"] if m]

    return dtc_codes


def merge_with_master_dtc(extracted_codes: Dict[str, Dict], master_file: Path) -> Tuple[int, int]:
    """
    Merge extracted DTC codes with the master DTC file.

    Args:
        extracted_codes: Dictionary of extracted DTC codes.
        master_file: Path to master DTC codes file.

    Returns:
        Tuple of (new_codes_added, existing_codes_updated).
    """
    if not master_file.exists():
        logger.warning(f"Master DTC file not found: {master_file}")
        return 0, 0

    master_data = load_json(master_file)
    if not master_data or "codes" not in master_data:
        logger.error("Invalid master DTC file format")
        return 0, 0

    # Create lookup for existing codes
    existing_codes = {c["code"]: c for c in master_data["codes"]}

    new_count = 0
    updated_count = 0

    for code, extracted_data in extracted_codes.items():
        if code in existing_codes:
            # Update existing code with NHTSA references
            existing = existing_codes[code]

            # Add NHTSA references if not present
            if "nhtsa_recalls" not in existing:
                existing["nhtsa_recalls"] = extracted_data["recall_references"]
                existing["nhtsa_complaints"] = extracted_data["complaint_references"]
                updated_count += 1
        else:
            # Add new code
            new_code = {
                "code": code,
                "description_en": f"Code extracted from NHTSA data (components: {', '.join(extracted_data['related_components'][:3])})",
                "description_hu": None,
                "category": get_category_from_code(code),
                "severity": get_severity_from_code(code),
                "system": get_system_from_code(code),
                "is_generic": code[1] == "0",
                "symptoms": [],
                "possible_causes": [],
                "diagnostic_steps": [],
                "related_codes": [],
                "source": "nhtsa",
                "nhtsa_recalls": extracted_data["recall_references"],
                "nhtsa_complaints": extracted_data["complaint_references"],
                "translation_status": "pending",
            }
            master_data["codes"].append(new_code)
            new_count += 1

    # Update metadata
    master_data["metadata"]["last_nhtsa_sync"] = datetime.now(timezone.utc).isoformat()
    master_data["metadata"]["total_codes"] = len(master_data["codes"])

    # Sort codes
    master_data["codes"].sort(key=lambda x: x["code"])

    # Save updated master file
    save_json(master_data, master_file)

    return new_count, updated_count


# =============================================================================
# Main Sync Functions
# =============================================================================


async def sync_recalls(
    client: NHTSASyncClient,
    makes: List[str],
    start_year: int,
    end_year: int,
) -> List[Dict[str, Any]]:
    """
    Sync recalls for specified makes and years.

    Args:
        client: NHTSA API client.
        makes: List of vehicle makes.
        start_year: Start year.
        end_year: End year (inclusive).

    Returns:
        List of all recall records.
    """
    all_recalls = []

    for make in makes:
        logger.info(f"Fetching recalls for {make}...")

        for year in range(start_year, end_year + 1):
            # Get models for this make/year
            models = await client.get_models_for_make(make, year)

            if not models:
                logger.debug(f"No models found for {make} {year}")
                continue

            # Limit to top 5 popular models to avoid excessive requests
            popular_models = models[:5]

            for model in popular_models:
                recalls = await client.get_recalls(make, model, year)

                if recalls:
                    all_recalls.extend(recalls)
                    logger.info(f"  {make} {model} {year}: {len(recalls)} recalls")

    return all_recalls


async def sync_complaints(
    client: NHTSASyncClient,
    makes: List[str],
    start_year: int,
    end_year: int,
) -> List[Dict[str, Any]]:
    """
    Sync complaints for specified makes and years.

    Args:
        client: NHTSA API client.
        makes: List of vehicle makes.
        start_year: Start year.
        end_year: End year (inclusive).

    Returns:
        List of all complaint records.
    """
    all_complaints = []

    for make in makes:
        logger.info(f"Fetching complaints for {make}...")

        for year in range(start_year, end_year + 1):
            # Get models for this make/year
            models = await client.get_models_for_make(make, year)

            if not models:
                continue

            # Limit to top 5 popular models
            popular_models = models[:5]

            for model in popular_models:
                complaints = await client.get_complaints(make, model, year)

                if complaints:
                    all_complaints.extend(complaints)
                    logger.info(f"  {make} {model} {year}: {len(complaints)} complaints")

    return all_complaints


# =============================================================================
# CLI Entry Point
# =============================================================================


async def main():
    """Main entry point for NHTSA sync."""
    parser = argparse.ArgumentParser(
        description="Sync vehicle recalls and complaints from NHTSA API"
    )
    parser.add_argument(
        "--recalls",
        action="store_true",
        help="Fetch recalls only",
    )
    parser.add_argument(
        "--complaints",
        action="store_true",
        help="Fetch complaints only",
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
        default=POPULAR_MAKES,
        help=f"Vehicle makes to fetch. Default: {', '.join(POPULAR_MAKES)}",
    )
    parser.add_argument(
        "--merge-dtc",
        action="store_true",
        help="Merge extracted DTC codes with master file",
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

    # Default to fetching both if neither specified
    fetch_recalls = args.recalls or (not args.recalls and not args.complaints)
    fetch_complaints = args.complaints or (not args.recalls and not args.complaints)

    logger.info("=" * 60)
    logger.info("NHTSA Data Sync")
    logger.info("=" * 60)
    logger.info(f"Makes: {', '.join(args.makes)}")
    logger.info(f"Years: {start_year}-{end_year}")
    logger.info(f"Fetch recalls: {fetch_recalls}")
    logger.info(f"Fetch complaints: {fetch_complaints}")
    logger.info("=" * 60)

    client = NHTSASyncClient(delay=REQUEST_DELAY)

    try:
        all_recalls = []
        all_complaints = []

        # Fetch recalls
        if fetch_recalls:
            all_recalls = await sync_recalls(client, args.makes, start_year, end_year)

            # Save recalls
            recalls_data = {
                "metadata": {
                    "synced_at": datetime.now(timezone.utc).isoformat(),
                    "makes": args.makes,
                    "year_range": f"{start_year}-{end_year}",
                    "total_recalls": len(all_recalls),
                },
                "recalls": all_recalls,
            }
            save_json(recalls_data, RECALLS_FILE)

        # Fetch complaints
        if fetch_complaints:
            all_complaints = await sync_complaints(client, args.makes, start_year, end_year)

            # Save complaints
            complaints_data = {
                "metadata": {
                    "synced_at": datetime.now(timezone.utc).isoformat(),
                    "makes": args.makes,
                    "year_range": f"{start_year}-{end_year}",
                    "total_complaints": len(all_complaints),
                },
                "complaints": all_complaints,
            }
            save_json(complaints_data, COMPLAINTS_FILE)

        # Extract and save DTC codes
        extracted_codes = extract_all_dtc_codes(all_recalls, all_complaints)

        if extracted_codes:
            extracted_data = {
                "metadata": {
                    "extracted_at": datetime.now(timezone.utc).isoformat(),
                    "total_codes": len(extracted_codes),
                    "from_recalls": len(all_recalls),
                    "from_complaints": len(all_complaints),
                },
                "codes": list(extracted_codes.values()),
            }
            save_json(extracted_data, EXTRACTED_DTC_FILE)
            logger.info(f"Extracted {len(extracted_codes)} unique DTC codes")

        # Merge with master DTC file if requested
        if args.merge_dtc and extracted_codes:
            new_count, updated_count = merge_with_master_dtc(extracted_codes, MASTER_DTC_FILE)
            logger.info(f"Merged with master: {new_count} new codes, {updated_count} updated")

        # Print summary
        print("\n" + "=" * 60)
        print("SYNC SUMMARY")
        print("=" * 60)
        print(f"Total recalls fetched: {len(all_recalls)}")
        print(f"Total complaints fetched: {len(all_complaints)}")
        print(f"Unique DTC codes extracted: {len(extracted_codes)}")
        print(f"\nOutput files:")
        print(f"  Recalls: {RECALLS_FILE}")
        print(f"  Complaints: {COMPLAINTS_FILE}")
        print(f"  Extracted DTC: {EXTRACTED_DTC_FILE}")

        if extracted_codes:
            print(f"\nTop extracted DTC codes:")
            # Sort by reference count
            sorted_codes = sorted(
                extracted_codes.values(),
                key=lambda x: len(x["recall_references"]) + len(x["complaint_references"]),
                reverse=True,
            )
            for code_data in sorted_codes[:10]:
                total_refs = len(code_data["recall_references"]) + len(code_data["complaint_references"])
                print(f"  {code_data['code']}: {total_refs} references")

        print("=" * 60)
        logger.info("Sync completed successfully!")

    except Exception as e:
        logger.error(f"Sync failed: {e}")
        raise

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
