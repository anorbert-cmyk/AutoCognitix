#!/usr/bin/env python3
"""
DTC Database Export Utility for AutoCognitix

Exports DTC (Diagnostic Trouble Code) data to multiple formats:
- JSON format (all_dtc_codes.json) - Full data with nested structures
- CSV format (dtc_codes.csv) - Tabular data for spreadsheets
- SQLite format (dtc_codes.sqlite) - Portable single-file database

Features:
- Includes Hungarian translations
- Includes related codes and symptoms
- Supports filtering by category, severity, source
- Progress tracking with tqdm
- Comprehensive logging

Usage:
    # Export all formats
    python scripts/export/export_dtc_database.py --all

    # Export specific format
    python scripts/export/export_dtc_database.py --json
    python scripts/export/export_dtc_database.py --csv
    python scripts/export/export_dtc_database.py --sqlite

    # With filtering
    python scripts/export/export_dtc_database.py --all --category P
    python scripts/export/export_dtc_database.py --all --severity high
    python scripts/export/export_dtc_database.py --all --translated-only

    # Custom output directory
    python scripts/export/export_dtc_database.py --all --output /path/to/output
"""

import argparse
import csv
import json
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    logger.warning("tqdm not installed. Progress bars will be disabled.")

    def tqdm(iterable, **kwargs):
        """Fallback tqdm that just returns the iterable."""
        return iterable

# Directories
DATA_DIR = PROJECT_ROOT / "data"
DTC_DATA_DIR = DATA_DIR / "dtc_codes"
EXPORT_DIR = DATA_DIR / "exports"

# Category mapping
CATEGORY_MAP = {
    "P": "powertrain",
    "B": "body",
    "C": "chassis",
    "U": "network",
}


class DTCExporter:
    """Handles exporting DTC codes to various formats."""

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        category_filter: Optional[List[str]] = None,
        severity_filter: Optional[str] = None,
        source_filter: Optional[str] = None,
        translated_only: bool = False,
        include_related: bool = True,
        include_symptoms: bool = True,
    ):
        """
        Initialize the DTC exporter.

        Args:
            output_dir: Output directory for exports
            category_filter: Filter by DTC categories (P, B, C, U)
            severity_filter: Filter by severity level
            source_filter: Filter by data source
            translated_only: Export only codes with Hungarian translations
            include_related: Include related DTC codes
            include_symptoms: Include symptom information
        """
        self.output_dir = output_dir or EXPORT_DIR
        self.category_filter = category_filter
        self.severity_filter = severity_filter
        self.source_filter = source_filter
        self.translated_only = translated_only
        self.include_related = include_related
        self.include_symptoms = include_symptoms

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Statistics
        self.stats = {
            "total_loaded": 0,
            "total_filtered": 0,
            "by_category": {},
            "by_severity": {},
            "translated": 0,
            "untranslated": 0,
        }

    def load_dtc_data(self) -> List[Dict[str, Any]]:
        """
        Load DTC data from all available sources.

        Returns:
            List of DTC code dictionaries
        """
        logger.info("Loading DTC data...")
        all_codes: Dict[str, Dict[str, Any]] = {}

        # Priority order for data sources
        data_sources = [
            ("all_codes_merged.json", "merged"),
            ("generic_codes.json", "generic"),
            ("klavkarr_codes.json", "klavkarr"),
            ("mytrile_codes.json", "mytrile"),
            ("obd_codes_com.json", "obd_codes_com"),
        ]

        for filename, source_name in data_sources:
            file_path = DTC_DATA_DIR / filename
            if file_path.exists():
                logger.info(f"  Loading {filename}...")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # Handle different data structures
                    codes = []
                    if isinstance(data, dict):
                        if "codes" in data:
                            codes = data["codes"]
                        elif "dtc_codes" in data:
                            codes = data["dtc_codes"]
                        else:
                            # Dictionary with code as key
                            for code, info in data.items():
                                if isinstance(info, dict):
                                    info["code"] = code
                                    codes.append(info)
                    elif isinstance(data, list):
                        codes = data

                    # Merge into all_codes
                    for code_data in codes:
                        code = code_data.get("code", "")
                        if not code:
                            continue

                        if code not in all_codes:
                            all_codes[code] = code_data.copy()
                            all_codes[code].setdefault("sources", [])
                            if source_name not in all_codes[code]["sources"]:
                                all_codes[code]["sources"].append(source_name)
                        else:
                            # Merge data from multiple sources
                            existing = all_codes[code]
                            for key, value in code_data.items():
                                if key == "sources":
                                    continue
                                # Prefer non-empty values
                                if value and not existing.get(key):
                                    existing[key] = value
                                # Merge arrays
                                elif isinstance(value, list) and isinstance(existing.get(key), list):
                                    for item in value:
                                        if item not in existing[key]:
                                            existing[key].append(item)
                            if source_name not in existing["sources"]:
                                existing["sources"].append(source_name)

                    logger.info(f"    Loaded {len(codes)} codes from {filename}")

                except json.JSONDecodeError as e:
                    logger.error(f"    Error parsing {filename}: {e}")
                except Exception as e:
                    logger.error(f"    Error loading {filename}: {e}")

        # Load translation cache if available
        translation_cache_path = DTC_DATA_DIR / "translation_cache.json"
        if translation_cache_path.exists():
            logger.info("  Loading translation cache...")
            try:
                with open(translation_cache_path, 'r', encoding='utf-8') as f:
                    translations = json.load(f)

                for code, translation in translations.items():
                    if code in all_codes:
                        if isinstance(translation, dict):
                            if translation.get("description_hu"):
                                all_codes[code]["description_hu"] = translation["description_hu"]
                        elif isinstance(translation, str) and translation:
                            all_codes[code]["description_hu"] = translation

                logger.info(f"    Applied {len(translations)} translations")
            except Exception as e:
                logger.warning(f"    Error loading translations: {e}")

        # Convert to list and sort
        dtc_list = sorted(all_codes.values(), key=lambda x: x.get("code", ""))

        self.stats["total_loaded"] = len(dtc_list)
        logger.info(f"Total DTC codes loaded: {len(dtc_list)}")

        return dtc_list

    def filter_codes(self, codes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply filters to the DTC code list.

        Args:
            codes: List of DTC code dictionaries

        Returns:
            Filtered list of DTC codes
        """
        filtered = []

        for code_data in codes:
            code = code_data.get("code", "")

            # Category filter
            if self.category_filter:
                code_category = code[0].upper() if code else ""
                if code_category not in self.category_filter:
                    continue

            # Severity filter
            if self.severity_filter:
                severity = code_data.get("severity", "medium").lower()
                if severity != self.severity_filter.lower():
                    continue

            # Source filter
            if self.source_filter:
                sources = code_data.get("sources", [])
                if self.source_filter not in sources:
                    continue

            # Translation filter
            if self.translated_only:
                if not code_data.get("description_hu"):
                    continue

            filtered.append(code_data)

            # Update statistics
            category = code[0].upper() if code else "Unknown"
            self.stats["by_category"][category] = self.stats["by_category"].get(category, 0) + 1

            severity = code_data.get("severity", "medium")
            self.stats["by_severity"][severity] = self.stats["by_severity"].get(severity, 0) + 1

            if code_data.get("description_hu"):
                self.stats["translated"] += 1
            else:
                self.stats["untranslated"] += 1

        self.stats["total_filtered"] = len(filtered)

        logger.info(f"Filtered DTC codes: {len(filtered)} (from {len(codes)})")
        if self.category_filter:
            logger.info(f"  Category filter: {self.category_filter}")
        if self.severity_filter:
            logger.info(f"  Severity filter: {self.severity_filter}")
        if self.translated_only:
            logger.info("  Translated only: Yes")

        return filtered

    def prepare_code_data(self, code_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare code data for export with all required fields.

        Args:
            code_data: Raw code data dictionary

        Returns:
            Prepared code data dictionary
        """
        code = code_data.get("code", "")

        # Determine category from code
        category_letter = code[0].upper() if code else ""
        category = CATEGORY_MAP.get(category_letter, "unknown")

        # Build prepared data
        prepared = {
            "code": code,
            "description_en": code_data.get("description_en") or code_data.get("description", ""),
            "description_hu": code_data.get("description_hu", ""),
            "category": category,
            "category_code": category_letter,
            "severity": code_data.get("severity", "medium"),
            "is_generic": code_data.get("is_generic", code[1] == "0" if len(code) > 1 else True),
            "system": code_data.get("system", ""),
            "sources": code_data.get("sources", []),
        }

        # Add symptoms if requested
        if self.include_symptoms:
            prepared["symptoms"] = code_data.get("symptoms", [])
            prepared["possible_causes"] = code_data.get("possible_causes", code_data.get("causes", []))
            prepared["diagnostic_steps"] = code_data.get("diagnostic_steps", [])

        # Add related codes if requested
        if self.include_related:
            prepared["related_codes"] = code_data.get("related_codes", code_data.get("related", []))

        # Add manufacturer-specific info
        prepared["applicable_makes"] = code_data.get("applicable_makes", [])
        prepared["manufacturer_code"] = code_data.get("manufacturer_code", "")

        return prepared

    def export_to_json(self, codes: List[Dict[str, Any]]) -> Path:
        """
        Export DTC codes to JSON format.

        Args:
            codes: List of DTC code dictionaries

        Returns:
            Path to exported file
        """
        logger.info("Exporting to JSON format...")

        # Prepare codes
        prepared_codes = []
        for code_data in tqdm(codes, desc="Preparing JSON data", disable=not HAS_TQDM):
            prepared_codes.append(self.prepare_code_data(code_data))

        # Build export data
        export_data = {
            "export_info": {
                "export_time": datetime.now().isoformat(),
                "source": "AutoCognitix",
                "version": "2.0",
                "total_codes": len(prepared_codes),
                "statistics": self.stats,
            },
            "filters_applied": {
                "category": self.category_filter,
                "severity": self.severity_filter,
                "source": self.source_filter,
                "translated_only": self.translated_only,
            },
            "codes": prepared_codes,
        }

        # Write JSON file
        output_file = self.output_dir / "all_dtc_codes.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)

        file_size = output_file.stat().st_size / 1024
        logger.info(f"JSON export complete: {output_file} ({file_size:.1f} KB)")

        return output_file

    def export_to_csv(self, codes: List[Dict[str, Any]]) -> Path:
        """
        Export DTC codes to CSV format.

        Args:
            codes: List of DTC code dictionaries

        Returns:
            Path to exported file
        """
        logger.info("Exporting to CSV format...")

        # Define CSV columns
        columns = [
            "code",
            "description_en",
            "description_hu",
            "category",
            "category_code",
            "severity",
            "is_generic",
            "system",
            "sources",
        ]

        if self.include_symptoms:
            columns.extend(["symptoms", "possible_causes", "diagnostic_steps"])

        if self.include_related:
            columns.append("related_codes")

        columns.extend(["applicable_makes", "manufacturer_code"])

        # Write CSV file
        output_file = self.output_dir / "dtc_codes.csv"
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
            writer.writeheader()

            for code_data in tqdm(codes, desc="Writing CSV rows", disable=not HAS_TQDM):
                prepared = self.prepare_code_data(code_data)

                # Convert lists to JSON strings for CSV
                row = {}
                for col in columns:
                    value = prepared.get(col, "")
                    if isinstance(value, (list, dict)):
                        value = json.dumps(value, ensure_ascii=False)
                    elif isinstance(value, bool):
                        value = "true" if value else "false"
                    row[col] = value

                writer.writerow(row)

        file_size = output_file.stat().st_size / 1024
        logger.info(f"CSV export complete: {output_file} ({file_size:.1f} KB)")

        return output_file

    def export_to_sqlite(self, codes: List[Dict[str, Any]]) -> Path:
        """
        Export DTC codes to SQLite database.

        Args:
            codes: List of DTC code dictionaries

        Returns:
            Path to exported file
        """
        logger.info("Exporting to SQLite format...")

        output_file = self.output_dir / "dtc_codes.sqlite"

        # Remove existing file
        if output_file.exists():
            output_file.unlink()

        conn = sqlite3.connect(output_file)
        cursor = conn.cursor()

        # Create main DTC codes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dtc_codes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT UNIQUE NOT NULL,
                description_en TEXT,
                description_hu TEXT,
                category TEXT,
                category_code TEXT,
                severity TEXT DEFAULT 'medium',
                is_generic INTEGER DEFAULT 1,
                system TEXT,
                sources TEXT,
                applicable_makes TEXT,
                manufacturer_code TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create symptoms table if needed
        if self.include_symptoms:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS dtc_symptoms (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dtc_code TEXT NOT NULL,
                    symptom TEXT NOT NULL,
                    FOREIGN KEY (dtc_code) REFERENCES dtc_codes(code)
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS dtc_causes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dtc_code TEXT NOT NULL,
                    cause TEXT NOT NULL,
                    FOREIGN KEY (dtc_code) REFERENCES dtc_codes(code)
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS dtc_diagnostic_steps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dtc_code TEXT NOT NULL,
                    step_order INTEGER,
                    step_text TEXT NOT NULL,
                    FOREIGN KEY (dtc_code) REFERENCES dtc_codes(code)
                )
            ''')

        # Create related codes table if needed
        if self.include_related:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS dtc_related (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dtc_code TEXT NOT NULL,
                    related_code TEXT NOT NULL,
                    FOREIGN KEY (dtc_code) REFERENCES dtc_codes(code)
                )
            ''')

        # Create metadata table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS export_metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')

        # Insert metadata
        cursor.execute(
            'INSERT INTO export_metadata VALUES (?, ?)',
            ("export_time", datetime.now().isoformat())
        )
        cursor.execute(
            'INSERT INTO export_metadata VALUES (?, ?)',
            ("source", "AutoCognitix")
        )
        cursor.execute(
            'INSERT INTO export_metadata VALUES (?, ?)',
            ("version", "2.0")
        )
        cursor.execute(
            'INSERT INTO export_metadata VALUES (?, ?)',
            ("total_codes", str(len(codes)))
        )

        # Insert DTC codes
        for code_data in tqdm(codes, desc="Inserting SQLite rows", disable=not HAS_TQDM):
            prepared = self.prepare_code_data(code_data)
            code = prepared["code"]

            cursor.execute('''
                INSERT OR REPLACE INTO dtc_codes
                (code, description_en, description_hu, category, category_code,
                 severity, is_generic, system, sources, applicable_makes, manufacturer_code)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                code,
                prepared["description_en"],
                prepared["description_hu"],
                prepared["category"],
                prepared["category_code"],
                prepared["severity"],
                1 if prepared["is_generic"] else 0,
                prepared["system"],
                json.dumps(prepared["sources"], ensure_ascii=False),
                json.dumps(prepared["applicable_makes"], ensure_ascii=False),
                prepared["manufacturer_code"],
            ))

            # Insert symptoms
            if self.include_symptoms:
                for symptom in prepared.get("symptoms", []):
                    cursor.execute(
                        'INSERT INTO dtc_symptoms (dtc_code, symptom) VALUES (?, ?)',
                        (code, symptom)
                    )

                for cause in prepared.get("possible_causes", []):
                    cursor.execute(
                        'INSERT INTO dtc_causes (dtc_code, cause) VALUES (?, ?)',
                        (code, cause)
                    )

                for i, step in enumerate(prepared.get("diagnostic_steps", [])):
                    cursor.execute(
                        'INSERT INTO dtc_diagnostic_steps (dtc_code, step_order, step_text) VALUES (?, ?, ?)',
                        (code, i + 1, step)
                    )

            # Insert related codes
            if self.include_related:
                for related_code in prepared.get("related_codes", []):
                    cursor.execute(
                        'INSERT INTO dtc_related (dtc_code, related_code) VALUES (?, ?)',
                        (code, related_code)
                    )

        # Create indexes for better query performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_dtc_code ON dtc_codes(code)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_dtc_category ON dtc_codes(category)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_dtc_severity ON dtc_codes(severity)')

        if self.include_symptoms:
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_symptoms_code ON dtc_symptoms(dtc_code)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_causes_code ON dtc_causes(dtc_code)')

        if self.include_related:
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_related_code ON dtc_related(dtc_code)')

        conn.commit()
        conn.close()

        file_size = output_file.stat().st_size / 1024
        logger.info(f"SQLite export complete: {output_file} ({file_size:.1f} KB)")

        return output_file

    def print_statistics(self) -> None:
        """Print export statistics."""
        logger.info("=" * 50)
        logger.info("EXPORT STATISTICS")
        logger.info("=" * 50)
        logger.info(f"Total codes loaded: {self.stats['total_loaded']}")
        logger.info(f"Total codes after filtering: {self.stats['total_filtered']}")
        logger.info("")
        logger.info("By category:")
        for cat, count in sorted(self.stats["by_category"].items()):
            cat_name = CATEGORY_MAP.get(cat, "Unknown")
            logger.info(f"  {cat} ({cat_name}): {count}")
        logger.info("")
        logger.info("By severity:")
        for sev, count in sorted(self.stats["by_severity"].items()):
            logger.info(f"  {sev}: {count}")
        logger.info("")
        logger.info(f"Translated (Hungarian): {self.stats['translated']}")
        logger.info(f"Untranslated: {self.stats['untranslated']}")
        logger.info("=" * 50)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Export DTC codes to multiple formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python export_dtc_database.py --all
    python export_dtc_database.py --json --csv
    python export_dtc_database.py --all --category P,B
    python export_dtc_database.py --all --severity high
    python export_dtc_database.py --all --translated-only
        """
    )

    # Format options
    format_group = parser.add_argument_group("Export Formats")
    format_group.add_argument(
        "--json",
        action="store_true",
        help="Export to JSON format (all_dtc_codes.json)",
    )
    format_group.add_argument(
        "--csv",
        action="store_true",
        help="Export to CSV format (dtc_codes.csv)",
    )
    format_group.add_argument(
        "--sqlite",
        action="store_true",
        help="Export to SQLite format (dtc_codes.sqlite)",
    )
    format_group.add_argument(
        "--all",
        action="store_true",
        help="Export to all formats",
    )

    # Filter options
    filter_group = parser.add_argument_group("Filtering Options")
    filter_group.add_argument(
        "--category",
        type=str,
        help="Filter by DTC category (P=powertrain, B=body, C=chassis, U=network). Comma-separated for multiple.",
    )
    filter_group.add_argument(
        "--severity",
        type=str,
        choices=["low", "medium", "high", "critical"],
        help="Filter by severity level",
    )
    filter_group.add_argument(
        "--source",
        type=str,
        help="Filter by data source (e.g., generic, klavkarr, mytrile)",
    )
    filter_group.add_argument(
        "--translated-only",
        action="store_true",
        help="Export only codes with Hungarian translations",
    )

    # Content options
    content_group = parser.add_argument_group("Content Options")
    content_group.add_argument(
        "--no-symptoms",
        action="store_true",
        help="Exclude symptom information",
    )
    content_group.add_argument(
        "--no-related",
        action="store_true",
        help="Exclude related codes",
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output directory (default: data/exports/)",
    )
    output_group.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse category filter
    category_filter = None
    if args.category:
        category_filter = [c.strip().upper() for c in args.category.split(",")]

    # Setup output directory
    output_dir = None
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Create exporter
    exporter = DTCExporter(
        output_dir=output_dir,
        category_filter=category_filter,
        severity_filter=args.severity,
        source_filter=args.source,
        translated_only=args.translated_only,
        include_related=not args.no_related,
        include_symptoms=not args.no_symptoms,
    )

    # Load and filter data
    codes = exporter.load_dtc_data()
    filtered_codes = exporter.filter_codes(codes)

    if not filtered_codes:
        logger.warning("No DTC codes match the specified filters!")
        sys.exit(1)

    # Default to --all if no specific format is selected
    if not any([args.json, args.csv, args.sqlite, args.all]):
        args.all = True

    # Export to requested formats
    exported_files = []

    try:
        if args.json or args.all:
            exported_files.append(exporter.export_to_json(filtered_codes))

        if args.csv or args.all:
            exported_files.append(exporter.export_to_csv(filtered_codes))

        if args.sqlite or args.all:
            exported_files.append(exporter.export_to_sqlite(filtered_codes))

        # Print statistics
        exporter.print_statistics()

        # Summary
        logger.info("")
        logger.info("Exported files:")
        for f in exported_files:
            logger.info(f"  {f}")

        logger.info("")
        logger.info("Export completed successfully!")

    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise


if __name__ == "__main__":
    main()
