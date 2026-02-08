#!/usr/bin/env python3
"""
UCI Automobile Dataset Import Script for AutoCognitix.

Downloads and processes the UCI Machine Learning Repository's Automobile
dataset containing specifications for 205 vehicles with 26 attributes.

Source: https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data

Features:
- Download with retry logic and exponential backoff
- Parse CSV with missing value handling ('?' marker)
- Convert to structured JSON with normalized field names
- Proper data type conversion (int, float, string)
- Hungarian translations for common automotive terms
- Neo4j-ready format with node structure
- Summary statistics output

Dataset Attributes (26 columns):
1. symboling: -3 to +3 (insurance risk rating)
2. normalized-losses: continuous from 65 to 256
3. make: alfa-romeo, audi, bmw, etc.
4. fuel-type: diesel, gas
5. aspiration: std, turbo
6. num-of-doors: four, two
7. body-style: hardtop, wagon, sedan, hatchback, convertible
8. drive-wheels: 4wd, fwd, rwd
9. engine-location: front, rear
10. wheel-base: continuous
11. length: continuous
12. width: continuous
13. height: continuous
14. curb-weight: continuous
15. engine-type: dohc, dohcv, l, ohc, ohcf, ohcv, rotor
16. num-of-cylinders: eight, five, four, six, three, twelve, two
17. engine-size: continuous
18. fuel-system: 1bbl, 2bbl, 4bbl, idi, mfi, mpfi, spdi, spfi
19. bore: continuous
20. stroke: continuous
21. compression-ratio: continuous
22. horsepower: continuous
23. peak-rpm: continuous
24. city-mpg: continuous
25. highway-mpg: continuous
26. price: continuous

Usage:
    python scripts/import_uci_automobile.py                  # Download and process
    python scripts/import_uci_automobile.py --dry-run        # Preview only
    python scripts/import_uci_automobile.py -v               # Verbose mode
    python scripts/import_uci_automobile.py --force          # Overwrite existing
"""

import argparse
import hashlib
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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

# UCI Dataset URL
UCI_DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
)
UCI_NAMES_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.names"
)

# Output paths
DATA_DIR = PROJECT_ROOT / "data" / "uci"
OUTPUT_FILE = DATA_DIR / "automobile_specs.json"
RAW_DATA_FILE = DATA_DIR / "imports-85.data"
STATS_FILE = DATA_DIR / "import_stats.json"

# Retry configuration
MAX_RETRIES = 3
RETRY_BACKOFF_FACTOR = 2.0
RETRY_STATUS_CODES = [429, 500, 502, 503, 504]

# Column definitions (0-indexed)
COLUMN_NAMES = [
    "symboling",
    "normalized_losses",
    "make",
    "fuel_type",
    "aspiration",
    "num_of_doors",
    "body_style",
    "drive_wheels",
    "engine_location",
    "wheel_base",
    "length",
    "width",
    "height",
    "curb_weight",
    "engine_type",
    "num_of_cylinders",
    "engine_size",
    "fuel_system",
    "bore",
    "stroke",
    "compression_ratio",
    "horsepower",
    "peak_rpm",
    "city_mpg",
    "highway_mpg",
    "price",
]

# Column types for parsing
COLUMN_TYPES: Dict[str, str] = {
    "symboling": "int",
    "normalized_losses": "float",
    "make": "string",
    "fuel_type": "string",
    "aspiration": "string",
    "num_of_doors": "string",  # "two", "four"
    "body_style": "string",
    "drive_wheels": "string",
    "engine_location": "string",
    "wheel_base": "float",
    "length": "float",
    "width": "float",
    "height": "float",
    "curb_weight": "int",
    "engine_type": "string",
    "num_of_cylinders": "string",  # "four", "six", etc.
    "engine_size": "int",
    "fuel_system": "string",
    "bore": "float",
    "stroke": "float",
    "compression_ratio": "float",
    "horsepower": "float",
    "peak_rpm": "float",
    "city_mpg": "int",
    "highway_mpg": "int",
    "price": "float",
}

# Hungarian translations for common terms
HUNGARIAN_TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "fuel_type": {
        "diesel": "dizel",
        "gas": "benzin",
    },
    "aspiration": {
        "std": "szivomotor",
        "turbo": "turbo",
    },
    "num_of_doors": {
        "two": "ketajtos",
        "four": "negyajtos",
    },
    "body_style": {
        "hardtop": "hardtop",
        "wagon": "kombi",
        "sedan": "sedan",
        "hatchback": "ferdehatu",
        "convertible": "kabrio",
    },
    "drive_wheels": {
        "4wd": "osszkerekezes",
        "fwd": "elsokerekezes",
        "rwd": "hatsokerekezes",
    },
    "engine_location": {
        "front": "elso",
        "rear": "hatso",
    },
    "engine_type": {
        "dohc": "dohc",
        "dohcv": "dohcv",
        "l": "soros",
        "ohc": "ohc",
        "ohcf": "ohcf",
        "ohcv": "ohcv",
        "rotor": "wankel",
    },
    "num_of_cylinders": {
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "eight": "8",
        "twelve": "12",
    },
}

# Cylinder count to integer mapping
CYLINDER_MAP: Dict[str, int] = {
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "eight": 8,
    "twelve": 12,
}

# Door count to integer mapping
DOOR_MAP: Dict[str, int] = {
    "two": 2,
    "four": 4,
}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class VehicleSpec:
    """Represents a single vehicle specification from the UCI dataset."""

    # Identifiers
    id: str
    make: str

    # Risk rating
    symboling: int  # -3 to +3
    normalized_losses: Optional[float]

    # Basic info
    fuel_type: str
    aspiration: str
    num_of_doors: Optional[int]
    body_style: str
    drive_wheels: str
    engine_location: str

    # Dimensions (cm/inches as per original)
    wheel_base: float
    length: float
    width: float
    height: float
    curb_weight: int  # kg

    # Engine specs
    engine_type: str
    num_of_cylinders: Optional[int]
    engine_size: int  # cc
    fuel_system: str
    bore: Optional[float]
    stroke: Optional[float]
    compression_ratio: float

    # Performance
    horsepower: Optional[float]
    peak_rpm: Optional[float]
    city_mpg: int
    highway_mpg: int

    # Price
    price: Optional[float]

    # Hungarian translations
    hu_fuel_type: str = ""
    hu_aspiration: str = ""
    hu_body_style: str = ""
    hu_drive_wheels: str = ""
    hu_engine_location: str = ""
    hu_engine_type: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "make": self.make,
            "symboling": self.symboling,
            "normalized_losses": self.normalized_losses,
            "fuel_type": self.fuel_type,
            "aspiration": self.aspiration,
            "num_of_doors": self.num_of_doors,
            "body_style": self.body_style,
            "drive_wheels": self.drive_wheels,
            "engine_location": self.engine_location,
            "wheel_base": self.wheel_base,
            "length": self.length,
            "width": self.width,
            "height": self.height,
            "curb_weight": self.curb_weight,
            "engine_type": self.engine_type,
            "num_of_cylinders": self.num_of_cylinders,
            "engine_size": self.engine_size,
            "fuel_system": self.fuel_system,
            "bore": self.bore,
            "stroke": self.stroke,
            "compression_ratio": self.compression_ratio,
            "horsepower": self.horsepower,
            "peak_rpm": self.peak_rpm,
            "city_mpg": self.city_mpg,
            "highway_mpg": self.highway_mpg,
            "price": self.price,
            "translations": {
                "hu": {
                    "fuel_type": self.hu_fuel_type,
                    "aspiration": self.hu_aspiration,
                    "body_style": self.hu_body_style,
                    "drive_wheels": self.hu_drive_wheels,
                    "engine_location": self.hu_engine_location,
                    "engine_type": self.hu_engine_type,
                }
            },
        }

    def to_neo4j_node(self) -> Dict[str, Any]:
        """Convert to Neo4j node format."""
        return {
            "labels": ["VehicleSpec", "UCIData"],
            "properties": {
                "id": self.id,
                "make": self.make,
                "symboling": self.symboling,
                "normalized_losses": self.normalized_losses,
                "fuel_type": self.fuel_type,
                "aspiration": self.aspiration,
                "num_of_doors": self.num_of_doors,
                "body_style": self.body_style,
                "drive_wheels": self.drive_wheels,
                "engine_location": self.engine_location,
                "wheel_base": self.wheel_base,
                "length": self.length,
                "width": self.width,
                "height": self.height,
                "curb_weight": self.curb_weight,
                "engine_type": self.engine_type,
                "num_of_cylinders": self.num_of_cylinders,
                "engine_size": self.engine_size,
                "fuel_system": self.fuel_system,
                "bore": self.bore,
                "stroke": self.stroke,
                "compression_ratio": self.compression_ratio,
                "horsepower": self.horsepower,
                "peak_rpm": self.peak_rpm,
                "city_mpg": self.city_mpg,
                "highway_mpg": self.highway_mpg,
                "price": self.price,
                "hu_fuel_type": self.hu_fuel_type,
                "hu_aspiration": self.hu_aspiration,
                "hu_body_style": self.hu_body_style,
                "hu_drive_wheels": self.hu_drive_wheels,
                "hu_engine_location": self.hu_engine_location,
                "hu_engine_type": self.hu_engine_type,
                "source": "uci_automobile",
            },
        }


@dataclass
class ImportStats:
    """Statistics for the import process."""

    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    missing_values: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    vehicles_by_make: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    vehicles_by_body_style: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    vehicles_by_fuel_type: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    price_stats: Dict[str, float] = field(default_factory=dict)
    horsepower_stats: Dict[str, float] = field(default_factory=dict)
    engine_size_stats: Dict[str, float] = field(default_factory=dict)
    download_time_seconds: float = 0.0
    parse_time_seconds: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0

    def add_vehicle(self, vehicle: VehicleSpec) -> None:
        """Update statistics with a new vehicle."""
        self.valid_records += 1
        self.vehicles_by_make[vehicle.make] += 1
        self.vehicles_by_body_style[vehicle.body_style] += 1
        self.vehicles_by_fuel_type[vehicle.fuel_type] += 1

    def add_missing(self, field_name: str) -> None:
        """Track missing value for a field."""
        self.missing_values[field_name] += 1

    def calculate_stats(self, vehicles: List[VehicleSpec]) -> None:
        """Calculate aggregate statistics."""
        prices = [v.price for v in vehicles if v.price is not None]
        horsepowers = [v.horsepower for v in vehicles if v.horsepower is not None]
        engine_sizes = [v.engine_size for v in vehicles]

        if prices:
            self.price_stats = {
                "min": min(prices),
                "max": max(prices),
                "avg": sum(prices) / len(prices),
                "count": len(prices),
            }

        if horsepowers:
            self.horsepower_stats = {
                "min": min(horsepowers),
                "max": max(horsepowers),
                "avg": sum(horsepowers) / len(horsepowers),
                "count": len(horsepowers),
            }

        if engine_sizes:
            self.engine_size_stats = {
                "min": min(engine_sizes),
                "max": max(engine_sizes),
                "avg": sum(engine_sizes) / len(engine_sizes),
                "count": len(engine_sizes),
            }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        elapsed = self.end_time - self.start_time if self.end_time else 0
        return {
            "summary": {
                "total_records": self.total_records,
                "valid_records": self.valid_records,
                "invalid_records": self.invalid_records,
                "elapsed_seconds": round(elapsed, 2),
                "download_time_seconds": round(self.download_time_seconds, 2),
                "parse_time_seconds": round(self.parse_time_seconds, 2),
            },
            "missing_values": dict(self.missing_values),
            "vehicles_by_make": dict(
                sorted(
                    self.vehicles_by_make.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            ),
            "vehicles_by_body_style": dict(self.vehicles_by_body_style),
            "vehicles_by_fuel_type": dict(self.vehicles_by_fuel_type),
            "price_stats": self.price_stats,
            "horsepower_stats": self.horsepower_stats,
            "engine_size_stats": self.engine_size_stats,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# =============================================================================
# HTTP Client with Retry
# =============================================================================


def create_session() -> requests.Session:
    """Create requests session with retry logic."""
    session = requests.Session()

    retry_strategy = Retry(
        total=MAX_RETRIES,
        backoff_factor=RETRY_BACKOFF_FACTOR,
        status_forcelist=RETRY_STATUS_CODES,
        allowed_methods=["GET"],
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session


def download_data(url: str, session: requests.Session) -> Tuple[str, float]:
    """
    Download data from URL with retry logic.

    Args:
        url: URL to download from.
        session: Requests session with retry configured.

    Returns:
        Tuple of (content, elapsed_seconds).

    Raises:
        requests.RequestException: If download fails after retries.
    """
    logger.info(f"Downloading from: {url}")
    start_time = time.time()

    response = session.get(url, timeout=30)
    response.raise_for_status()

    elapsed = time.time() - start_time
    content = response.text

    logger.info(
        f"Downloaded {len(content)} bytes in {elapsed:.2f}s "
        f"(HTTP {response.status_code})"
    )

    return content, elapsed


# =============================================================================
# Data Parsing
# =============================================================================


def parse_value(
    value: str,
    col_name: str,
    col_type: str,
    stats: ImportStats
) -> Optional[Union[int, float, str]]:
    """
    Parse a single value with type conversion.

    Args:
        value: Raw string value from CSV.
        col_name: Column name for error reporting.
        col_type: Expected type ('int', 'float', 'string').
        stats: Statistics object to track missing values.

    Returns:
        Parsed value or None if missing/invalid.
    """
    value = value.strip()

    # Handle missing values
    if value == "?" or value == "":
        stats.add_missing(col_name)
        return None

    try:
        if col_type == "int":
            return int(float(value))  # Handle "123.0" format
        elif col_type == "float":
            return float(value)
        else:
            return value.lower()
    except ValueError as e:
        logger.warning(f"Failed to parse {col_name}='{value}' as {col_type}: {e}")
        stats.add_missing(col_name)
        return None


def generate_vehicle_id(row_data: Dict[str, Any], row_index: int) -> str:
    """Generate unique ID for a vehicle based on its attributes."""
    key_parts = [
        str(row_data.get("make", "")),
        str(row_data.get("body_style", "")),
        str(row_data.get("engine_size", "")),
        str(row_data.get("horsepower", "")),
        str(row_index),
    ]
    key_string = "-".join(key_parts)
    hash_hex = hashlib.md5(key_string.encode()).hexdigest()[:8]
    return f"uci-{row_data.get('make', 'unknown')}-{hash_hex}"


def get_hungarian_translation(field: str, value: str) -> str:
    """Get Hungarian translation for a field value."""
    translations = HUNGARIAN_TRANSLATIONS.get(field, {})
    return translations.get(value, value)


def parse_row(
    row: str,
    row_index: int,
    stats: ImportStats
) -> Optional[VehicleSpec]:
    """
    Parse a single CSV row into a VehicleSpec.

    Args:
        row: Raw CSV row string.
        row_index: Row number for ID generation.
        stats: Statistics object.

    Returns:
        VehicleSpec or None if row is invalid.
    """
    # Split by comma
    values = row.strip().split(",")

    if len(values) != len(COLUMN_NAMES):
        logger.warning(
            f"Row {row_index}: Expected {len(COLUMN_NAMES)} columns, "
            f"got {len(values)}"
        )
        stats.invalid_records += 1
        return None

    # Parse all values
    row_data: Dict[str, Any] = {}
    for col_name, value in zip(COLUMN_NAMES, values):
        col_type = COLUMN_TYPES[col_name]
        parsed = parse_value(value, col_name, col_type, stats)
        row_data[col_name] = parsed

    # Convert door count
    doors_str = row_data.get("num_of_doors")
    num_doors = DOOR_MAP.get(doors_str) if doors_str else None

    # Convert cylinder count
    cylinders_str = row_data.get("num_of_cylinders")
    num_cylinders = CYLINDER_MAP.get(cylinders_str) if cylinders_str else None

    # Generate ID
    vehicle_id = generate_vehicle_id(row_data, row_index)

    # Get make (required field)
    make = row_data.get("make")
    if not make:
        logger.warning(f"Row {row_index}: Missing required field 'make'")
        stats.invalid_records += 1
        return None

    # Build VehicleSpec
    try:
        vehicle = VehicleSpec(
            id=vehicle_id,
            make=make,
            symboling=row_data.get("symboling") or 0,
            normalized_losses=row_data.get("normalized_losses"),
            fuel_type=row_data.get("fuel_type") or "unknown",
            aspiration=row_data.get("aspiration") or "unknown",
            num_of_doors=num_doors,
            body_style=row_data.get("body_style") or "unknown",
            drive_wheels=row_data.get("drive_wheels") or "unknown",
            engine_location=row_data.get("engine_location") or "front",
            wheel_base=row_data.get("wheel_base") or 0.0,
            length=row_data.get("length") or 0.0,
            width=row_data.get("width") or 0.0,
            height=row_data.get("height") or 0.0,
            curb_weight=row_data.get("curb_weight") or 0,
            engine_type=row_data.get("engine_type") or "unknown",
            num_of_cylinders=num_cylinders,
            engine_size=row_data.get("engine_size") or 0,
            fuel_system=row_data.get("fuel_system") or "unknown",
            bore=row_data.get("bore"),
            stroke=row_data.get("stroke"),
            compression_ratio=row_data.get("compression_ratio") or 0.0,
            horsepower=row_data.get("horsepower"),
            peak_rpm=row_data.get("peak_rpm"),
            city_mpg=row_data.get("city_mpg") or 0,
            highway_mpg=row_data.get("highway_mpg") or 0,
            price=row_data.get("price"),
            # Hungarian translations
            hu_fuel_type=get_hungarian_translation(
                "fuel_type", row_data.get("fuel_type") or ""
            ),
            hu_aspiration=get_hungarian_translation(
                "aspiration", row_data.get("aspiration") or ""
            ),
            hu_body_style=get_hungarian_translation(
                "body_style", row_data.get("body_style") or ""
            ),
            hu_drive_wheels=get_hungarian_translation(
                "drive_wheels", row_data.get("drive_wheels") or ""
            ),
            hu_engine_location=get_hungarian_translation(
                "engine_location", row_data.get("engine_location") or ""
            ),
            hu_engine_type=get_hungarian_translation(
                "engine_type", row_data.get("engine_type") or ""
            ),
        )
        return vehicle
    except Exception as e:
        logger.warning(f"Row {row_index}: Failed to create VehicleSpec: {e}")
        stats.invalid_records += 1
        return None


def parse_dataset(content: str, stats: ImportStats) -> List[VehicleSpec]:
    """
    Parse the entire dataset.

    Args:
        content: Raw CSV content.
        stats: Statistics object.

    Returns:
        List of VehicleSpec objects.
    """
    logger.info("Parsing dataset...")
    start_time = time.time()

    vehicles: List[VehicleSpec] = []
    lines = content.strip().split("\n")
    stats.total_records = len(lines)

    for i, line in enumerate(lines):
        if not line.strip():
            continue

        vehicle = parse_row(line, i, stats)
        if vehicle:
            vehicles.append(vehicle)
            stats.add_vehicle(vehicle)

    stats.parse_time_seconds = time.time() - start_time
    stats.calculate_stats(vehicles)

    logger.info(
        f"Parsed {stats.valid_records}/{stats.total_records} valid records "
        f"in {stats.parse_time_seconds:.2f}s"
    )

    return vehicles


# =============================================================================
# Output
# =============================================================================


def save_json(
    vehicles: List[VehicleSpec],
    output_path: Path,
    stats: ImportStats,
) -> None:
    """
    Save vehicles to JSON file.

    Args:
        vehicles: List of vehicle specs.
        output_path: Path to output file.
        stats: Import statistics.
    """
    logger.info(f"Saving {len(vehicles)} vehicles to {output_path}")

    output_data = {
        "metadata": {
            "source": "UCI Machine Learning Repository",
            "source_url": UCI_DATA_URL,
            "description": "Automobile dataset with 205 vehicle specifications",
            "attributes_count": 26,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "valid_records": stats.valid_records,
            "total_records": stats.total_records,
        },
        "vehicles": [v.to_dict() for v in vehicles],
        "neo4j_nodes": [v.to_neo4j_node() for v in vehicles],
        "statistics": stats.to_dict(),
    }

    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    file_size = output_path.stat().st_size
    logger.info(f"Saved {file_size:,} bytes to {output_path}")


def save_raw_data(content: str, output_path: Path) -> None:
    """Save raw downloaded data for caching."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.debug(f"Saved raw data to {output_path}")


def print_summary(stats: ImportStats, vehicles: List[VehicleSpec]) -> None:
    """Print summary statistics to console."""
    print("\n" + "=" * 60)
    print("UCI AUTOMOBILE DATASET IMPORT SUMMARY")
    print("=" * 60)

    print(f"\nRecords:")
    print(f"  Total:   {stats.total_records}")
    print(f"  Valid:   {stats.valid_records}")
    print(f"  Invalid: {stats.invalid_records}")

    print(f"\nVehicles by Make (Top 10):")
    sorted_makes = sorted(
        stats.vehicles_by_make.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:10]
    for make, count in sorted_makes:
        print(f"  {make.capitalize():20} {count:3}")

    print(f"\nVehicles by Body Style:")
    for style, count in stats.vehicles_by_body_style.items():
        print(f"  {style.capitalize():20} {count:3}")

    print(f"\nVehicles by Fuel Type:")
    for fuel, count in stats.vehicles_by_fuel_type.items():
        print(f"  {fuel.capitalize():20} {count:3}")

    if stats.price_stats:
        print(f"\nPrice Statistics (USD):")
        print(f"  Min:     ${stats.price_stats['min']:,.0f}")
        print(f"  Max:     ${stats.price_stats['max']:,.0f}")
        print(f"  Average: ${stats.price_stats['avg']:,.0f}")
        print(f"  Records: {stats.price_stats['count']}")

    if stats.horsepower_stats:
        print(f"\nHorsepower Statistics:")
        print(f"  Min:     {stats.horsepower_stats['min']:.0f} HP")
        print(f"  Max:     {stats.horsepower_stats['max']:.0f} HP")
        print(f"  Average: {stats.horsepower_stats['avg']:.0f} HP")

    if stats.engine_size_stats:
        print(f"\nEngine Size Statistics (cc):")
        print(f"  Min:     {stats.engine_size_stats['min']:.0f}")
        print(f"  Max:     {stats.engine_size_stats['max']:.0f}")
        print(f"  Average: {stats.engine_size_stats['avg']:.0f}")

    print(f"\nMissing Values by Field:")
    sorted_missing = sorted(
        stats.missing_values.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    for field, count in sorted_missing:
        pct = (count / stats.total_records) * 100
        print(f"  {field:25} {count:3} ({pct:.1f}%)")

    print(f"\nTiming:")
    print(f"  Download: {stats.download_time_seconds:.2f}s")
    print(f"  Parse:    {stats.parse_time_seconds:.2f}s")
    total = stats.end_time - stats.start_time
    print(f"  Total:    {total:.2f}s")

    print("\n" + "=" * 60)


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Import UCI Automobile dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/import_uci_automobile.py
    python scripts/import_uci_automobile.py --dry-run
    python scripts/import_uci_automobile.py --force -v
        """,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate only, don't save files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output file",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use cached raw data if available",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize stats
    stats = ImportStats()
    stats.start_time = time.time()

    try:
        # Check if output already exists
        if OUTPUT_FILE.exists() and not args.force and not args.dry_run:
            logger.error(
                f"Output file already exists: {OUTPUT_FILE}\n"
                "Use --force to overwrite or --dry-run to preview"
            )
            return 1

        # Download or use cache
        if args.use_cache and RAW_DATA_FILE.exists():
            logger.info(f"Using cached data from {RAW_DATA_FILE}")
            with open(RAW_DATA_FILE, "r", encoding="utf-8") as f:
                content = f.read()
            stats.download_time_seconds = 0.0
        else:
            session = create_session()
            content, download_time = download_data(UCI_DATA_URL, session)
            stats.download_time_seconds = download_time

            # Cache raw data
            if not args.dry_run:
                save_raw_data(content, RAW_DATA_FILE)

        # Parse dataset
        vehicles = parse_dataset(content, stats)

        if not vehicles:
            logger.error("No valid vehicles parsed from dataset")
            return 1

        # Save output
        stats.end_time = time.time()

        if args.dry_run:
            logger.info("Dry run - not saving files")
        else:
            save_json(vehicles, OUTPUT_FILE, stats)

            # Save stats separately
            with open(STATS_FILE, "w", encoding="utf-8") as f:
                json.dump(stats.to_dict(), f, indent=2)
            logger.info(f"Saved statistics to {STATS_FILE}")

        # Print summary
        print_summary(stats, vehicles)

        return 0

    except requests.RequestException as e:
        logger.error(f"Download failed: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Import failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
