#!/usr/bin/env python3
"""
NHTSA FLAT_CMPL.txt Parser for AutoCognitix.

Parses the NHTSA complaint flat file (TAB-separated, ~2.1M records)
and outputs normalized JSON files split by year range.

Features:
- Streaming line-by-line reader (no full file load into memory)
- Configurable year filtering (default: 2000+)
- Normalized snake_case field names matching existing data format
- Output split by year range (e.g., 2000-2005.json, 2006-2010.json)
- Checkpoint logging every 100K records
- Detailed statistics at completion
- Handles latin-1/cp1252 encoding from NHTSA files

NHTSA Flat File Format (49 TAB-separated fields, no header):
  [0]  CMPLID       - Sequential complaint ID
  [1]  ODESSION     - ODI number
  [2]  MFR_NAME     - Manufacturer full name
  [3]  MAKETXT      - Vehicle make (uppercase)
  [4]  MODELTXT     - Vehicle model
  [5]  YEARTXT      - Model year
  [6]  CRASH        - Crash involved (Y/N)
  [7]  FAILDATE     - Failure/incident date (YYYYMMDD)
  [8]  FIRE         - Fire involved (Y/N)
  [9]  INJURED      - Number of injuries
  [10] DEATHS       - Number of deaths
  [11] COMPDESC     - Component description
  [12] CITY         - City
  [13] STATE        - State code
  [14] VIN          - VIN (truncated to 11 chars)
  [15] DATEA        - Date complaint received (YYYYMMDD)
  [16] LDATE        - Date added to file (YYYYMMDD)
  [17] MILES        - Mileage at failure
  [18] OCCURENCES   - Number of occurrences
  [19] CDESCR       - Consumer description/summary
  [20] CMPL_TYPE    - Complaint type (EVOQ, IVOQ, CON, etc.)
  [21] POLICE_RPT   - Police report filed (Y/N)
  [22] PESSION      - Purchase date (YYYYMMDD)
  [23] MEDICAL_ATTN - Medical attention sought (Y/N)
  [24] VEHICLES_TOWED - Vehicle towed (Y/N)
  ...  [25-48] Additional/supplemental fields

Usage:
    python scripts/import_flat_complaints.py
    python scripts/import_flat_complaints.py --min-year 2010
    python scripts/import_flat_complaints.py --min-year 2000 --max-year 2024
    python scripts/import_flat_complaints.py --no-split  # Single output file
"""

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Project root
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

FLAT_FILE_PATH = PROJECT_ROOT / "data" / "nhtsa" / "flat_files" / "FLAT_CMPL.txt"
OUTPUT_DIR = PROJECT_ROOT / "data" / "nhtsa" / "complaints_flat"
CHECKPOINT_DIR = PROJECT_ROOT / "scripts" / "checkpoints"

# Year range bins for output splitting
YEAR_RANGES = [
    (2000, 2005),
    (2006, 2010),
    (2011, 2015),
    (2016, 2020),
    (2021, 2026),
]

# Checkpoint interval
CHECKPOINT_INTERVAL = 100_000

# Encoding: NHTSA files commonly use latin-1 or cp1252
FILE_ENCODING = "latin-1"

# Expected number of fields per record
EXPECTED_FIELDS = 49


# =============================================================================
# Field Mapping
# =============================================================================

def parse_date(raw: str) -> Optional[str]:
    """Parse YYYYMMDD date string to ISO format YYYY-MM-DD.

    Args:
        raw: Date string in YYYYMMDD format.

    Returns:
        ISO date string or None if invalid.
    """
    raw = raw.strip()
    if not raw or len(raw) != 8 or not raw.isdigit():
        return None
    try:
        year = int(raw[:4])
        month = int(raw[4:6])
        day = int(raw[6:8])
        if year < 1900 or year > 2030 or month < 1 or month > 12 or day < 1 or day > 31:
            return None
        return f"{year:04d}-{month:02d}-{day:02d}"
    except (ValueError, IndexError):
        return None


def safe_int(raw: str, default: int = 0) -> int:
    """Safely parse integer from string.

    Args:
        raw: String to parse.
        default: Default value on failure.

    Returns:
        Parsed integer or default.
    """
    raw = raw.strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        # Handle floats like "40000.0"
        try:
            return int(float(raw))
        except ValueError:
            return default


def parse_record(fields: List[str]) -> Optional[Dict[str, Any]]:
    """Parse a single TSV record into a normalized dictionary.

    Args:
        fields: List of 49 tab-separated field values.

    Returns:
        Normalized dictionary or None if record is invalid.
    """
    if len(fields) < 20:
        return None

    # Pad fields to expected length if short
    while len(fields) < EXPECTED_FIELDS:
        fields.append("")

    make = fields[3].strip()
    model = fields[4].strip()
    year_raw = fields[5].strip()
    summary = fields[19].strip()

    # Skip records with missing essential fields
    if not make or not model or not year_raw:
        return None

    # Parse model year
    model_year = safe_int(year_raw, 0)
    if model_year == 0 or model_year == 9999:
        return None

    # Build normalized record matching existing complaint format
    record = {
        "complaint_id": safe_int(fields[0]),
        "odi_number": fields[1].strip(),
        "manufacturer": fields[2].strip(),
        "make": make,
        "model": model,
        "model_year": model_year,
        "crash": fields[6].strip().upper() == "Y",
        "fire": fields[8].strip().upper() == "Y",
        "injuries": safe_int(fields[9]),
        "deaths": safe_int(fields[10]),
        "component": fields[11].strip() or "UNKNOWN",
        "city": fields[12].strip(),
        "state": fields[13].strip(),
        "vin": fields[14].strip() or None,
        "date_received": parse_date(fields[15]),
        "date_added": parse_date(fields[16]),
        "date_of_incident": parse_date(fields[7]),
        "mileage": safe_int(fields[17]) if fields[17].strip() else None,
        "occurrences": safe_int(fields[18]) if fields[18].strip() else None,
        "summary": summary,
        "complaint_type": fields[20].strip() if len(fields) > 20 else None,
        "police_report": fields[21].strip().upper() == "Y" if len(fields) > 21 else False,
        "medical_attention": fields[23].strip().upper() == "Y" if len(fields) > 23 else False,
        "vehicle_towed": fields[24].strip().upper() == "Y" if len(fields) > 24 else False,
    }

    return record


# =============================================================================
# Statistics Tracker
# =============================================================================


class Stats:
    """Track import statistics."""

    def __init__(self) -> None:
        self.total_lines = 0
        self.parsed_ok = 0
        self.skipped_year = 0
        self.skipped_empty = 0
        self.parse_errors = 0
        self.field_count_errors = 0
        self.encoding_errors = 0

        self.makes: Dict[str, int] = defaultdict(int)
        self.years: Dict[int, int] = defaultdict(int)
        self.components: Dict[str, int] = defaultdict(int)
        self.crashes = 0
        self.fires = 0
        self.injuries = 0
        self.deaths = 0

        self.start_time = time.monotonic()
        self.end_time = 0.0

    def add_record(self, record: Dict[str, Any]) -> None:
        """Update stats with a parsed record."""
        self.parsed_ok += 1
        self.makes[record["make"]] += 1
        self.years[record["model_year"]] += 1
        self.components[record["component"]] += 1
        if record["crash"]:
            self.crashes += 1
        if record["fire"]:
            self.fires += 1
        self.injuries += record["injuries"]
        self.deaths += record["deaths"]

    def to_dict(self) -> Dict[str, Any]:
        """Export stats as dictionary."""
        elapsed = self.end_time - self.start_time if self.end_time else 0

        top_makes = dict(
            sorted(self.makes.items(), key=lambda x: x[1], reverse=True)[:30]
        )
        top_components = dict(
            sorted(self.components.items(), key=lambda x: x[1], reverse=True)[:25]
        )

        return {
            "parsing": {
                "total_lines": self.total_lines,
                "parsed_ok": self.parsed_ok,
                "skipped_year_filter": self.skipped_year,
                "skipped_empty_fields": self.skipped_empty,
                "parse_errors": self.parse_errors,
                "field_count_errors": self.field_count_errors,
                "encoding_errors": self.encoding_errors,
            },
            "data": {
                "unique_makes": len(self.makes),
                "unique_years": len(self.years),
                "unique_components": len(self.components),
                "year_range": (
                    f"{min(self.years)}-{max(self.years)}" if self.years else "N/A"
                ),
            },
            "by_make": top_makes,
            "by_year": dict(sorted(self.years.items())),
            "top_components": top_components,
            "safety": {
                "crashes": self.crashes,
                "fires": self.fires,
                "injuries": self.injuries,
                "deaths": self.deaths,
            },
            "performance": {
                "elapsed_seconds": round(elapsed, 2),
                "records_per_second": (
                    round(self.parsed_ok / elapsed, 0) if elapsed > 0 else 0
                ),
            },
        }

    def print_summary(self) -> None:
        """Print human-readable summary."""
        elapsed = self.end_time - self.start_time

        print("\n" + "=" * 70)
        print("NHTSA FLAT FILE IMPORT SUMMARY")
        print("=" * 70)

        print(f"\n{'Parsing:':<30}")
        print(f"  Total lines read:          {self.total_lines:>12,}")
        print(f"  Successfully parsed:       {self.parsed_ok:>12,}")
        print(f"  Skipped (year filter):     {self.skipped_year:>12,}")
        print(f"  Skipped (empty fields):    {self.skipped_empty:>12,}")
        print(f"  Parse errors:              {self.parse_errors:>12,}")
        print(f"  Field count errors:        {self.field_count_errors:>12,}")

        print(f"\n{'Data:':<30}")
        print(f"  Unique makes:              {len(self.makes):>12,}")
        print(f"  Unique components:         {len(self.components):>12,}")
        if self.years:
            print(f"  Year range:                {min(self.years)}-{max(self.years)}")

        print(f"\n{'Top 15 Makes:':<30}")
        top_makes = sorted(self.makes.items(), key=lambda x: x[1], reverse=True)[:15]
        for make, count in top_makes:
            print(f"  {make:<25} {count:>10,}")

        print(f"\n{'By Year (last 10):':<30}")
        recent_years = sorted(self.years.items())[-10:]
        for year, count in recent_years:
            print(f"  {year:<25} {count:>10,}")

        print(f"\n{'Safety Metrics:':<30}")
        print(f"  Crashes:                   {self.crashes:>12,}")
        print(f"  Fires:                     {self.fires:>12,}")
        print(f"  Injuries:                  {self.injuries:>12,}")
        print(f"  Deaths:                    {self.deaths:>12,}")

        print(f"\n{'Performance:':<30}")
        print(f"  Elapsed:                   {elapsed:>10.1f}s")
        if elapsed > 0:
            print(f"  Records/second:            {self.parsed_ok / elapsed:>10,.0f}")

        print("=" * 70)


# =============================================================================
# Year Range Bin Helper
# =============================================================================


def get_year_range_key(year: int, ranges: List[Tuple[int, int]]) -> Optional[str]:
    """Get the year range bin key for a given year.

    Args:
        year: Model year.
        ranges: List of (start, end) tuples.

    Returns:
        String key like '2000-2005' or None if not in any range.
    """
    for start, end in ranges:
        if start <= year <= end:
            return f"{start}-{end}"
    return None


# =============================================================================
# Main Parser
# =============================================================================


def parse_flat_file(
    input_path: Path,
    output_dir: Path,
    min_year: int = 2000,
    max_year: int = 2026,
    year_ranges: Optional[List[Tuple[int, int]]] = None,
    split_output: bool = True,
) -> Stats:
    """Parse NHTSA flat file and write JSON outputs.

    Reads the file line-by-line (streaming) to avoid loading the full
    1.5GB file into memory. Writes output in year-range JSON files.

    Args:
        input_path: Path to FLAT_CMPL.txt.
        output_dir: Directory for JSON output files.
        min_year: Minimum model year to include.
        max_year: Maximum model year to include.
        year_ranges: Year range bins for splitting output.
        split_output: If True, split into year-range files. If False, single file.

    Returns:
        Stats object with import statistics.
    """
    if year_ranges is None:
        year_ranges = YEAR_RANGES

    stats = Stats()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Buffers for year-range output (write in chunks to avoid memory pressure)
    buffers: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    buffer_counts: Dict[str, int] = defaultdict(int)
    BUFFER_FLUSH_SIZE = 50_000  # Flush to disk every 50K records per bin

    # Track written file handles for append mode
    files_started: set = set()

    def flush_buffer(range_key: str, final: bool = False) -> None:
        """Write buffered records to disk as JSON array segments.

        For the first write, starts a new JSON array.
        For subsequent writes, appends to the existing file.
        On final flush, closes the JSON array.
        """
        if not buffers[range_key]:
            return

        filepath = output_dir / f"{range_key}.json"

        if range_key not in files_started:
            # First write: start the JSON structure
            with open(filepath, "w", encoding="utf-8") as f:
                metadata = {
                    "source": "NHTSA FLAT_CMPL.txt",
                    "parsed_at": datetime.now(timezone.utc).isoformat(),
                    "year_range": range_key,
                    "format": "flat_file_import",
                }
                f.write('{\n  "metadata": ')
                json.dump(metadata, f, ensure_ascii=False)
                f.write(',\n  "complaints": [\n')
                # Write first batch
                for i, record in enumerate(buffers[range_key]):
                    prefix = "    " if i == 0 else ",\n    "
                    f.write(prefix)
                    json.dump(record, f, ensure_ascii=False)
            files_started.add(range_key)
        else:
            # Append mode
            with open(filepath, "a", encoding="utf-8") as f:
                for record in buffers[range_key]:
                    f.write(",\n    ")
                    json.dump(record, f, ensure_ascii=False)

        buffer_counts[range_key] += len(buffers[range_key])
        buffers[range_key] = []

    def close_all_files() -> None:
        """Close all JSON arrays properly."""
        for range_key in files_started:
            filepath = output_dir / f"{range_key}.json"
            with open(filepath, "a", encoding="utf-8") as f:
                f.write("\n  ]\n}\n")

    logger.info(f"Reading: {input_path}")
    logger.info(f"Output:  {output_dir}")
    logger.info(f"Year filter: {min_year}-{max_year}")
    logger.info(f"Split output: {split_output}")

    try:
        with open(input_path, "r", encoding=FILE_ENCODING, errors="replace") as f:
            for line in f:
                stats.total_lines += 1

                # Checkpoint logging
                if stats.total_lines % CHECKPOINT_INTERVAL == 0:
                    elapsed = time.monotonic() - stats.start_time
                    rate = stats.total_lines / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"Progress: {stats.total_lines:>10,} lines | "
                        f"{stats.parsed_ok:>10,} parsed | "
                        f"{rate:,.0f} lines/sec | "
                        f"elapsed: {elapsed:.0f}s"
                    )

                line = line.rstrip("\n\r")
                if not line:
                    continue

                fields = line.split("\t")

                # Basic field count validation
                if len(fields) < 20:
                    stats.field_count_errors += 1
                    continue

                try:
                    record = parse_record(fields)
                except Exception as e:
                    stats.parse_errors += 1
                    if stats.parse_errors <= 10:
                        logger.warning(
                            f"Parse error at line {stats.total_lines}: {e}"
                        )
                    continue

                if record is None:
                    stats.skipped_empty += 1
                    continue

                # Year filter
                if record["model_year"] < min_year or record["model_year"] > max_year:
                    stats.skipped_year += 1
                    continue

                # Track statistics
                stats.add_record(record)

                if split_output:
                    range_key = get_year_range_key(record["model_year"], year_ranges)
                    if range_key is None:
                        # Year is in filter range but not in any bin - put in closest
                        range_key = f"{min_year}-{max_year}"

                    buffers[range_key].append(record)

                    # Flush buffer if large enough
                    if len(buffers[range_key]) >= BUFFER_FLUSH_SIZE:
                        flush_buffer(range_key)
                else:
                    range_key = f"{min_year}-{max_year}"
                    buffers[range_key].append(record)
                    if len(buffers[range_key]) >= BUFFER_FLUSH_SIZE:
                        flush_buffer(range_key)

        # Final flush
        for range_key in list(buffers.keys()):
            flush_buffer(range_key, final=True)

        # Close JSON arrays
        close_all_files()

    except UnicodeDecodeError as e:
        stats.encoding_errors += 1
        logger.error(f"Encoding error: {e}")

    stats.end_time = time.monotonic()

    # Log output file sizes
    logger.info("\nOutput files:")
    for range_key in sorted(files_started):
        filepath = output_dir / f"{range_key}.json"
        size_mb = filepath.stat().st_size / (1024 * 1024)
        count = buffer_counts[range_key]
        logger.info(f"  {filepath.name}: {count:,} records ({size_mb:.1f} MB)")

    # Save statistics
    stats_path = output_dir / "flat_import_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "source": str(input_path),
                    "parsed_at": datetime.now(timezone.utc).isoformat(),
                    "year_filter": f"{min_year}-{max_year}",
                    "split_output": split_output,
                },
                "statistics": stats.to_dict(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info(f"Statistics saved to: {stats_path}")

    return stats


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Parse NHTSA FLAT_CMPL.txt into JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/import_flat_complaints.py
    python scripts/import_flat_complaints.py --min-year 2010
    python scripts/import_flat_complaints.py --min-year 1995 --max-year 2026
    python scripts/import_flat_complaints.py --no-split
    python scripts/import_flat_complaints.py --input data/nhtsa/flat_files/FLAT_CMPL.txt
        """,
    )

    parser.add_argument(
        "--input",
        type=str,
        default=str(FLAT_FILE_PATH),
        help=f"Path to FLAT_CMPL.txt (default: {FLAT_FILE_PATH})",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help=f"Output directory (default: {OUTPUT_DIR})",
    )

    parser.add_argument(
        "--min-year",
        type=int,
        default=2000,
        help="Minimum model year to include (default: 2000)",
    )

    parser.add_argument(
        "--max-year",
        type=int,
        default=2026,
        help="Maximum model year to include (default: 2026)",
    )

    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Output a single JSON file instead of splitting by year range",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose/debug logging",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        logger.error(
            "Download it first:\n"
            "  mkdir -p data/nhtsa/flat_files\n"
            '  curl -L "https://static.nhtsa.gov/odi/ffdd/cmpl/FLAT_CMPL.zip" '
            "-o data/nhtsa/flat_files/FLAT_CMPL.zip\n"
            "  cd data/nhtsa/flat_files && unzip FLAT_CMPL.zip"
        )
        return 1

    # Build year ranges dynamically if custom range
    year_ranges = []
    start = args.min_year
    while start <= args.max_year:
        end = min(start + 4, args.max_year)
        year_ranges.append((start, end))
        start = end + 1

    logger.info("=" * 60)
    logger.info("NHTSA Flat File Complaint Parser")
    logger.info("=" * 60)
    logger.info(f"Input:       {input_path}")
    logger.info(f"Output:      {output_dir}")
    logger.info(f"Year range:  {args.min_year}-{args.max_year}")
    logger.info(f"Year bins:   {', '.join(f'{s}-{e}' for s, e in year_ranges)}")
    logger.info(f"Split:       {not args.no_split}")
    logger.info("=" * 60)

    try:
        stats = parse_flat_file(
            input_path=input_path,
            output_dir=output_dir,
            min_year=args.min_year,
            max_year=args.max_year,
            year_ranges=year_ranges,
            split_output=not args.no_split,
        )

        stats.print_summary()

        if stats.parsed_ok > 0:
            logger.info("Import completed successfully!")
            return 0
        else:
            logger.warning("No records parsed - check input file and year filter")
            return 1

    except KeyboardInterrupt:
        logger.info("\nImport cancelled by user")
        return 130

    except Exception as e:
        logger.error(f"Import failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
