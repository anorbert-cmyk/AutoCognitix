#!/usr/bin/env python3
"""
DTC Data Merging Script for AutoCognitix

This script merges DTC codes from multiple sources into a unified dataset,
then updates PostgreSQL and Neo4j databases.

Data Sources:
- data/dtc_codes/mytrile_codes.json
- data/dtc_codes/klavkarr_codes.json
- data/dtc_codes/obd_codes_com.json
- data/dtc_codes/dtcbase_codes.json
- data/dtc_codes/troublecodes_net.json
- data/dtc_codes/generic_codes.json

Features:
- Deduplication by DTC code
- Intelligent merge strategy (keep best descriptions, combine symptoms/causes)
- Conflict resolution (prefer manufacturer-specific over generic)
- Source tracking for audit trail
- Batch database operations for efficiency
- Comprehensive logging and reporting

Usage:
    python scripts/merge_dtc_sources.py --all              # Merge and update all databases
    python scripts/merge_dtc_sources.py --merge-only       # Only merge to JSON
    python scripts/merge_dtc_sources.py --postgres         # Update PostgreSQL only
    python scripts/merge_dtc_sources.py --neo4j            # Update Neo4j only
    python scripts/merge_dtc_sources.py --dry-run          # Preview without changes
"""

import argparse
import hashlib
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils import (
    get_category_from_code,
    get_severity_from_code,
    get_sync_db_url,
    get_system_from_code,
    sanitize_text,
    setup_logging,
    validate_dtc_code,
)

# Configure logging
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = PROJECT_ROOT / "data" / "dtc_codes"
OUTPUT_FILE = DATA_DIR / "all_codes_merged.json"
REPORT_FILE = DATA_DIR / "merge_report.json"

# Source file configuration with priority (higher = preferred for conflicts)
SOURCE_FILES: Dict[str, Dict[str, Any]] = {
    "generic": {
        "path": DATA_DIR / "generic_codes.json",
        "priority": 1,  # Lowest priority (fallback)
        "description": "Generic OBD-II codes",
    },
    "klavkarr": {
        "path": DATA_DIR / "klavkarr_codes.json",
        "priority": 2,
        "description": "Klavkarr automotive database",
    },
    "obd_codes_com": {
        "path": DATA_DIR / "obd_codes_com.json",
        "priority": 3,
        "description": "OBD-Codes.com database",
    },
    "dtcbase": {
        "path": DATA_DIR / "dtcbase_codes.json",
        "priority": 4,
        "description": "DTCBase comprehensive database",
    },
    "troublecodes_net": {
        "path": DATA_DIR / "troublecodes_net.json",
        "priority": 5,
        "description": "TroubleCodes.net database",
    },
    "mytrile": {
        "path": DATA_DIR / "mytrile_codes.json",
        "priority": 6,  # Highest priority (most detailed)
        "description": "MyTrile curated codes with Hungarian translations",
    },
}

# Similarity threshold for description comparison
DESCRIPTION_SIMILARITY_THRESHOLD = 0.85

# Maximum length limits for security
MAX_DESCRIPTION_LENGTH = 2000
MAX_ARRAY_ITEMS = 50
MAX_CODE_LENGTH = 10


@dataclass
class MergeDecision:
    """Record of a merge decision for audit trail."""

    code: str
    decision_type: str  # "new", "merged", "conflict_resolved", "skipped"
    sources_merged: List[str]
    fields_updated: List[str] = field(default_factory=list)
    conflict_details: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class MergeReport:
    """Summary report of merge operation."""

    total_unique_codes: int = 0
    total_input_codes: int = 0
    sources_processed: List[str] = field(default_factory=list)
    codes_by_source: Dict[str, int] = field(default_factory=dict)
    codes_per_category: Dict[str, int] = field(default_factory=dict)
    conflicts_resolved: int = 0
    new_descriptions_added: int = 0
    symptoms_combined: int = 0
    causes_combined: int = 0
    skipped_invalid: int = 0
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_unique_codes": self.total_unique_codes,
            "total_input_codes": self.total_input_codes,
            "sources_processed": self.sources_processed,
            "codes_by_source": self.codes_by_source,
            "codes_per_category": self.codes_per_category,
            "conflicts_resolved": self.conflicts_resolved,
            "new_descriptions_added": self.new_descriptions_added,
            "symptoms_combined": self.symptoms_combined,
            "causes_combined": self.causes_combined,
            "skipped_invalid": self.skipped_invalid,
            "decisions_count": len(self.decisions),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


class DTCMerger:
    """Handles merging of DTC codes from multiple sources."""

    def __init__(self, dry_run: bool = False, batch_size: int = 100):
        """
        Initialize the merger.

        Args:
            dry_run: If True, don't write to databases.
            batch_size: Number of records to process per batch.
        """
        self.dry_run = dry_run
        self.batch_size = batch_size
        self.merged_codes: Dict[str, Dict[str, Any]] = {}
        self.report = MergeReport()
        self._engine = None

    def load_source(self, source_name: str) -> List[Dict[str, Any]]:
        """
        Load codes from a single source file with validation.

        Args:
            source_name: Key from SOURCE_FILES dict.

        Returns:
            List of validated code dictionaries.
        """
        source_config = SOURCE_FILES.get(source_name)
        if not source_config:
            logger.warning(f"Unknown source: {source_name}")
            return []

        file_path = source_config["path"]
        if not file_path.exists():
            logger.info(f"Source file not found (may not be scraped yet): {file_path}")
            return []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                # Security: Limit file size
                if len(content) > 50 * 1024 * 1024:  # 50MB limit
                    logger.error(f"Source file too large: {file_path}")
                    return []
                data = json.loads(content)

            codes = data.get("codes", [])
            validated = []

            for code_data in codes:
                validated_code = self._validate_and_sanitize(code_data, source_name)
                if validated_code:
                    validated.append(validated_code)

            logger.info(f"Loaded {len(validated)} valid codes from {source_name}")
            self.report.codes_by_source[source_name] = len(validated)
            self.report.total_input_codes += len(validated)

            return validated

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {file_path}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return []

    def _validate_and_sanitize(
        self, code_data: Dict[str, Any], source_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Validate and sanitize a single code entry.

        Args:
            code_data: Raw code dictionary from source.
            source_name: Name of the source for tracking.

        Returns:
            Sanitized code dictionary or None if invalid.
        """
        code = str(code_data.get("code", "")).upper().strip()

        # Validate code format
        if not code or len(code) > MAX_CODE_LENGTH:
            return None

        if not validate_dtc_code(code):
            return None

        # Get description
        description_en = code_data.get("description_en", "")
        if isinstance(description_en, str):
            description_en = sanitize_text(description_en, MAX_DESCRIPTION_LENGTH)
        else:
            description_en = ""

        # Skip codes without description
        if not description_en:
            return None

        # Sanitize Hungarian description
        description_hu = code_data.get("description_hu")
        if description_hu and isinstance(description_hu, str):
            description_hu = sanitize_text(description_hu, MAX_DESCRIPTION_LENGTH)
        else:
            description_hu = None

        # Sanitize arrays
        def sanitize_array(arr: Any) -> List[str]:
            if not isinstance(arr, list):
                return []
            result = []
            for item in arr[:MAX_ARRAY_ITEMS]:
                if isinstance(item, str):
                    sanitized = sanitize_text(item, 500)
                    if sanitized:
                        result.append(sanitized)
            return result

        return {
            "code": code,
            "description_en": description_en,
            "description_hu": description_hu,
            "category": get_category_from_code(code),
            "severity": code_data.get("severity") or get_severity_from_code(code),
            "system": code_data.get("system") or get_system_from_code(code),
            "is_generic": code[1] == "0",  # Second char 0 = generic
            "symptoms": sanitize_array(code_data.get("symptoms", [])),
            "possible_causes": sanitize_array(code_data.get("possible_causes", [])),
            "diagnostic_steps": sanitize_array(code_data.get("diagnostic_steps", [])),
            "related_codes": sanitize_array(code_data.get("related_codes", [])),
            "source": source_name,
            "manufacturer": code_data.get("manufacturer"),
            "translation_status": code_data.get("translation_status", "pending"),
        }

    def merge_all_sources(self) -> Dict[str, Dict[str, Any]]:
        """
        Load and merge all configured sources.

        Returns:
            Dictionary of merged codes keyed by code string.
        """
        logger.info("Starting merge from all sources...")

        # Load sources in priority order (lowest first)
        sorted_sources = sorted(
            SOURCE_FILES.items(), key=lambda x: x[1]["priority"]
        )

        for source_name, config in sorted_sources:
            codes = self.load_source(source_name)
            if codes:
                self.report.sources_processed.append(source_name)
                for code_data in codes:
                    self._merge_single_code(code_data, config["priority"])

        # Update category counts
        for code, data in self.merged_codes.items():
            category = data.get("category", "unknown")
            self.report.codes_per_category[category] = (
                self.report.codes_per_category.get(category, 0) + 1
            )

        self.report.total_unique_codes = len(self.merged_codes)
        logger.info(f"Merge complete: {self.report.total_unique_codes} unique codes")

        return self.merged_codes

    def _merge_single_code(self, new_code: Dict[str, Any], priority: int) -> None:
        """
        Merge a single code into the merged dataset.

        Args:
            new_code: Code dictionary to merge.
            priority: Source priority (higher = preferred).
        """
        code = new_code["code"]

        if code not in self.merged_codes:
            # New code - add directly
            new_code["sources"] = [new_code.pop("source", "unknown")]
            new_code["_priority"] = priority
            self.merged_codes[code] = new_code

            decision = MergeDecision(
                code=code,
                decision_type="new",
                sources_merged=new_code["sources"],
            )
            self.report.decisions.append(decision.__dict__)
            self.report.new_descriptions_added += 1
            return

        # Existing code - merge intelligently
        existing = self.merged_codes[code]
        source = new_code.pop("source", "unknown")

        # Track sources
        if source not in existing.get("sources", []):
            existing.setdefault("sources", []).append(source)

        decision = MergeDecision(
            code=code,
            decision_type="merged",
            sources_merged=existing["sources"],
            fields_updated=[],
        )

        # Merge descriptions - prefer longer, more detailed
        if self._should_update_description(existing, new_code, priority):
            if existing.get("description_en") != new_code.get("description_en"):
                existing["description_en"] = new_code["description_en"]
                decision.fields_updated.append("description_en")
                self.report.new_descriptions_added += 1

        # Merge Hungarian descriptions if missing or better
        if new_code.get("description_hu"):
            if not existing.get("description_hu") or (
                priority > existing.get("_priority", 0)
                and len(new_code.get("description_hu", "")) > len(existing.get("description_hu", ""))
            ):
                existing["description_hu"] = new_code["description_hu"]
                decision.fields_updated.append("description_hu")

        # Merge arrays - combine unique items
        for arr_field in ["symptoms", "possible_causes", "diagnostic_steps", "related_codes"]:
            if new_code.get(arr_field):
                existing_arr = set(existing.get(arr_field, []))
                new_items = set(new_code.get(arr_field, []))
                combined = existing_arr | new_items

                if len(combined) > len(existing_arr):
                    existing[arr_field] = list(combined)[:MAX_ARRAY_ITEMS]
                    decision.fields_updated.append(arr_field)

                    if arr_field == "symptoms":
                        self.report.symptoms_combined += len(new_items - existing_arr)
                    elif arr_field == "possible_causes":
                        self.report.causes_combined += len(new_items - existing_arr)

        # Update severity if new source has higher priority
        if priority > existing.get("_priority", 0):
            if new_code.get("severity") and new_code["severity"] != existing.get("severity"):
                existing["severity"] = new_code["severity"]
                decision.fields_updated.append("severity")

            if new_code.get("system") and new_code["system"] != existing.get("system"):
                existing["system"] = new_code["system"]
                decision.fields_updated.append("system")

        # Track manufacturer specificity
        if new_code.get("manufacturer") and not existing.get("manufacturer"):
            existing["manufacturer"] = new_code["manufacturer"]
            existing["is_generic"] = False
            decision.fields_updated.append("manufacturer")

        # Update priority
        existing["_priority"] = max(existing.get("_priority", 0), priority)

        if decision.fields_updated:
            self.report.decisions.append(decision.__dict__)

    def _should_update_description(
        self, existing: Dict[str, Any], new_code: Dict[str, Any], priority: int
    ) -> bool:
        """
        Determine if description should be updated.

        Criteria:
        - New description is significantly longer
        - New description has higher source priority
        - Descriptions are substantially different (not just minor variations)

        Args:
            existing: Current code data.
            new_code: New code data to potentially merge.
            priority: Source priority of new code.

        Returns:
            True if description should be updated.
        """
        existing_desc = existing.get("description_en", "")
        new_desc = new_code.get("description_en", "")

        if not new_desc:
            return False

        if not existing_desc:
            return True

        # Check similarity
        similarity = SequenceMatcher(None, existing_desc.lower(), new_desc.lower()).ratio()

        # If very similar, prefer longer one
        if similarity >= DESCRIPTION_SIMILARITY_THRESHOLD:
            return len(new_desc) > len(existing_desc) * 1.2

        # If different, check if new has higher priority
        existing_priority = existing.get("_priority", 0)
        if priority > existing_priority:
            return len(new_desc) >= len(existing_desc) * 0.8

        # Significantly longer new description wins
        return len(new_desc) > len(existing_desc) * 1.5

    def resolve_conflicts(self) -> None:
        """
        Resolve any remaining conflicts in merged data.

        This handles edge cases like:
        - Multiple significantly different descriptions
        - Conflicting severity levels
        """
        logger.info("Resolving conflicts...")

        for code, data in self.merged_codes.items():
            # Clean up internal tracking fields
            data.pop("_priority", None)

            # Ensure required fields have values
            if not data.get("category"):
                data["category"] = get_category_from_code(code)

            if not data.get("severity"):
                data["severity"] = get_severity_from_code(code)

            if not data.get("system"):
                data["system"] = get_system_from_code(code)

            # Ensure sources is a list
            if not isinstance(data.get("sources"), list):
                data["sources"] = [data.get("source", "unknown")]

            # Remove redundant source field
            data.pop("source", None)

            # Validate is_generic is boolean
            data["is_generic"] = bool(data.get("is_generic", code[1] == "0"))

    def save_merged_data(self) -> Path:
        """
        Save merged data to JSON file.

        Returns:
            Path to the output file.
        """
        if self.dry_run:
            logger.info("[DRY RUN] Would save merged data to %s", OUTPUT_FILE)
            return OUTPUT_FILE

        # Sort codes by code string
        sorted_codes = sorted(self.merged_codes.values(), key=lambda x: x["code"])

        # Calculate stats
        translated_count = sum(1 for c in sorted_codes if c.get("description_hu"))

        output_data = {
            "metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "total_codes": len(sorted_codes),
                "translated": translated_count,
                "sources": self.report.sources_processed,
            },
            "codes": sorted_codes,
        }

        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(sorted_codes)} codes to {OUTPUT_FILE}")
        return OUTPUT_FILE

    def save_report(self) -> Path:
        """
        Save merge report to JSON file.

        Returns:
            Path to the report file.
        """
        self.report.completed_at = datetime.now(timezone.utc).isoformat()

        if self.dry_run:
            logger.info("[DRY RUN] Would save report to %s", REPORT_FILE)
            return REPORT_FILE

        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            json.dump(self.report.to_dict(), f, ensure_ascii=False, indent=2)

        logger.info(f"Saved report to {REPORT_FILE}")
        return REPORT_FILE

    def update_postgres(self) -> Tuple[int, int, int]:
        """
        Update PostgreSQL database with merged codes.

        Returns:
            Tuple of (inserted, updated, skipped) counts.
        """
        if self.dry_run:
            logger.info("[DRY RUN] Would update PostgreSQL with %d codes", len(self.merged_codes))
            return (0, 0, len(self.merged_codes))

        from sqlalchemy import create_engine
        from sqlalchemy.orm import Session

        from backend.app.db.postgres.models import Base, DTCCode

        engine = create_engine(get_sync_db_url())
        Base.metadata.create_all(engine)

        inserted = 0
        updated = 0
        skipped = 0

        codes_list = list(self.merged_codes.values())

        with Session(engine) as session:
            try:
                for i in tqdm(
                    range(0, len(codes_list), self.batch_size),
                    desc="Updating PostgreSQL",
                ):
                    batch = codes_list[i : i + self.batch_size]

                    for code_data in batch:
                        code = code_data["code"]

                        # Use parameterized query (safe from SQL injection)
                        existing = session.query(DTCCode).filter(
                            DTCCode.code == code
                        ).first()

                        if existing:
                            # Update existing record
                            changed = False

                            # Update description if better
                            if len(code_data.get("description_en", "")) > len(
                                existing.description_en or ""
                            ):
                                existing.description_en = code_data["description_en"]
                                changed = True

                            # Update Hungarian if missing or better
                            if code_data.get("description_hu") and (
                                not existing.description_hu
                                or len(code_data["description_hu"])
                                > len(existing.description_hu)
                            ):
                                existing.description_hu = code_data["description_hu"]
                                changed = True

                            # Merge arrays
                            for arr_field in [
                                "symptoms",
                                "possible_causes",
                                "diagnostic_steps",
                                "related_codes",
                                "sources",
                            ]:
                                existing_arr = set(getattr(existing, arr_field, []) or [])
                                new_arr = set(code_data.get(arr_field, []))
                                combined = existing_arr | new_arr

                                if len(combined) > len(existing_arr):
                                    setattr(
                                        existing,
                                        arr_field,
                                        list(combined)[:MAX_ARRAY_ITEMS],
                                    )
                                    changed = True

                            if changed:
                                updated += 1
                            else:
                                skipped += 1
                        else:
                            # Insert new record
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
                                sources=code_data.get("sources", []),
                            )
                            session.add(dtc)
                            inserted += 1

                    # Commit each batch
                    session.commit()

            except Exception as e:
                session.rollback()
                logger.error(f"PostgreSQL update failed: {e}")
                raise

        logger.info(
            f"PostgreSQL: inserted={inserted}, updated={updated}, skipped={skipped}"
        )
        return (inserted, updated, skipped)

    def update_neo4j(self) -> Tuple[int, int]:
        """
        Update Neo4j database with merged codes.

        Returns:
            Tuple of (created, updated) counts.
        """
        if self.dry_run:
            logger.info("[DRY RUN] Would update Neo4j with %d codes", len(self.merged_codes))
            return (0, 0)

        from backend.app.db.neo4j_models import DTCNode, SymptomNode

        created = 0
        updated = 0

        codes_list = list(self.merged_codes.values())

        for code_data in tqdm(codes_list, desc="Updating Neo4j"):
            try:
                code = code_data["code"]
                existing = DTCNode.nodes.get_or_none(code=code)

                if existing:
                    # Update existing node
                    changed = False

                    if len(code_data.get("description_en", "")) > len(
                        existing.description_en or ""
                    ):
                        existing.description_en = code_data["description_en"]
                        changed = True

                    if code_data.get("description_hu") and (
                        not existing.description_hu
                        or len(code_data["description_hu"]) > len(existing.description_hu)
                    ):
                        existing.description_hu = code_data["description_hu"]
                        changed = True

                    if changed:
                        existing.save()
                        updated += 1
                else:
                    # Create new node
                    dtc = DTCNode(
                        code=code_data["code"],
                        description_en=code_data["description_en"],
                        description_hu=code_data.get("description_hu"),
                        category=code_data.get("category", "unknown"),
                        severity=code_data.get("severity", "medium"),
                        is_generic=str(code_data.get("is_generic", True)).lower(),
                        system=code_data.get("system", ""),
                    ).save()
                    created += 1

                    # Create symptom relationships
                    for symptom_name in code_data.get("symptoms", [])[:10]:
                        symptom = SymptomNode.nodes.get_or_none(name=symptom_name)
                        if not symptom:
                            symptom = SymptomNode(
                                name=symptom_name, description_hu=symptom_name
                            ).save()
                        if not dtc.causes.is_connected(symptom):
                            dtc.causes.connect(symptom, {"confidence": 0.7})

            except Exception as e:
                logger.warning(f"Neo4j error for {code_data['code']}: {e}")
                continue

        logger.info(f"Neo4j: created={created}, updated={updated}")
        return (created, updated)


def print_summary(report: MergeReport) -> None:
    """Print a human-readable summary of the merge operation."""
    print("\n" + "=" * 60)
    print("DTC MERGE SUMMARY")
    print("=" * 60)
    print(f"\nTotal unique codes after merge: {report.total_unique_codes}")
    print(f"Total input codes (with duplicates): {report.total_input_codes}")
    print(f"\nSources processed: {', '.join(report.sources_processed)}")
    print("\nCodes by source:")
    for source, count in sorted(report.codes_by_source.items()):
        print(f"  - {source}: {count}")
    print("\nCodes by category:")
    for category, count in sorted(report.codes_per_category.items()):
        print(f"  - {category}: {count}")
    print(f"\nConflicts resolved: {report.conflicts_resolved}")
    print(f"New descriptions added: {report.new_descriptions_added}")
    print(f"Symptoms combined: {report.symptoms_combined}")
    print(f"Causes combined: {report.causes_combined}")
    print(f"Invalid codes skipped: {report.skipped_invalid}")
    print("=" * 60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Merge DTC codes from multiple sources into PostgreSQL and Neo4j"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Merge and update all databases",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Only merge to JSON file, don't update databases",
    )
    parser.add_argument(
        "--postgres",
        action="store_true",
        help="Update PostgreSQL only (requires merged JSON)",
    )
    parser.add_argument(
        "--neo4j",
        action="store_true",
        help="Update Neo4j only (requires merged JSON)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing to files or databases",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for database operations (default: 100)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Default to --all if no specific action is selected
    if not any([args.all, args.merge_only, args.postgres, args.neo4j]):
        args.all = True

    setup_logging(args.verbose)

    try:
        merger = DTCMerger(dry_run=args.dry_run, batch_size=args.batch_size)

        # Merge from all sources
        if args.all or args.merge_only:
            merger.merge_all_sources()
            merger.resolve_conflicts()
            merger.save_merged_data()
            merger.save_report()
        else:
            # Load existing merged data for database-only updates
            if OUTPUT_FILE.exists():
                logger.info(f"Loading existing merged data from {OUTPUT_FILE}")
                with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for code_data in data.get("codes", []):
                    merger.merged_codes[code_data["code"]] = code_data
                logger.info(f"Loaded {len(merger.merged_codes)} codes")
            else:
                logger.error(f"Merged data file not found: {OUTPUT_FILE}")
                logger.error("Run with --all or --merge-only first")
                sys.exit(1)

        # Update databases
        if args.all or args.postgres:
            merger.update_postgres()

        if args.all or args.neo4j:
            merger.update_neo4j()

        # Print summary
        print_summary(merger.report)

        logger.info("Merge operation completed successfully!")

    except Exception as e:
        logger.error(f"Merge operation failed: {e}")
        raise


if __name__ == "__main__":
    main()
