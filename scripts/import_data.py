#!/usr/bin/env python3
"""
Comprehensive Import Script for AutoCognitix

Imports data from exported formats with merge and conflict resolution:
- JSON format (from export_data.py)
- CSV format (from export_data.py)
- SQLite database (from export_data.py)
- GraphML format (Neo4j graph import)
- Qdrant vectors restore

Conflict Resolution Strategies:
- skip: Skip existing records (default)
- overwrite: Replace existing records
- merge: Merge fields (keep existing, add new)
- newest: Keep record with latest timestamp

Usage:
    python scripts/import_data.py --json path/to/export.json              # Import from JSON
    python scripts/import_data.py --csv path/to/csv/directory             # Import from CSV
    python scripts/import_data.py --sqlite path/to/export.db              # Import from SQLite
    python scripts/import_data.py --graphml path/to/graph.graphml         # Import Neo4j graph

    # Conflict resolution:
    python scripts/import_data.py --json export.json --on-conflict skip       # Skip duplicates
    python scripts/import_data.py --json export.json --on-conflict overwrite  # Overwrite duplicates
    python scripts/import_data.py --json export.json --on-conflict merge      # Merge fields
    python scripts/import_data.py --json export.json --on-conflict newest     # Keep newest

    # Other options:
    python scripts/import_data.py --json export.json --dry-run                # Preview without changes
    python scripts/import_data.py --json export.json --validate               # Validate data only
"""

import argparse
import csv
import gzip
import json
import logging
import sqlite3
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Data directories
DATA_DIR = PROJECT_ROOT / "data"


class ConflictResolver:
    """Handles conflict resolution during import."""

    STRATEGIES = ["skip", "overwrite", "merge", "newest"]

    def __init__(self, strategy: str = "skip"):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}. Must be one of {self.STRATEGIES}")
        self.strategy = strategy
        self.stats = {
            "inserted": 0,
            "updated": 0,
            "skipped": 0,
            "merged": 0,
            "errors": 0,
        }

    def resolve(
        self,
        existing: Optional[Dict[str, Any]],
        incoming: Dict[str, Any],
        key_field: str = "code",
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Resolve conflict between existing and incoming records.

        Returns:
            Tuple of (action, resolved_data) where action is one of:
            - "insert": Insert as new record
            - "update": Update existing record
            - "skip": Skip this record
        """
        if existing is None:
            self.stats["inserted"] += 1
            return "insert", incoming

        if self.strategy == "skip":
            self.stats["skipped"] += 1
            return "skip", existing

        elif self.strategy == "overwrite":
            self.stats["updated"] += 1
            return "update", incoming

        elif self.strategy == "merge":
            merged = self._merge_records(existing, incoming)
            self.stats["merged"] += 1
            return "update", merged

        elif self.strategy == "newest":
            action, result = self._keep_newest(existing, incoming)
            if action == "update":
                self.stats["updated"] += 1
            else:
                self.stats["skipped"] += 1
            return action, result

        return "skip", existing

    def _merge_records(
        self,
        existing: Dict[str, Any],
        incoming: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Merge two records, keeping existing values and adding new fields."""
        merged = dict(existing)

        for key, value in incoming.items():
            # Skip if existing has a non-empty value
            existing_value = merged.get(key)
            if existing_value is None or existing_value == "" or existing_value == []:
                merged[key] = value
            elif isinstance(value, list) and isinstance(existing_value, list):
                # Merge lists (union)
                merged[key] = list(set(existing_value + value))

        return merged

    def _keep_newest(
        self,
        existing: Dict[str, Any],
        incoming: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        """Keep the record with the most recent timestamp."""
        existing_time = self._get_timestamp(existing)
        incoming_time = self._get_timestamp(incoming)

        if incoming_time and (not existing_time or incoming_time > existing_time):
            return "update", incoming
        return "skip", existing

    def _get_timestamp(self, record: Dict[str, Any]) -> Optional[datetime]:
        """Extract timestamp from a record."""
        for field in ["updated_at", "created_at", "timestamp"]:
            value = record.get(field)
            if value:
                if isinstance(value, datetime):
                    return value
                if isinstance(value, str):
                    try:
                        return datetime.fromisoformat(value.replace("Z", "+00:00"))
                    except ValueError:
                        pass
        return None

    def get_summary(self) -> str:
        """Get a summary of import statistics."""
        return (
            f"Inserted: {self.stats['inserted']}, "
            f"Updated: {self.stats['updated']}, "
            f"Merged: {self.stats['merged']}, "
            f"Skipped: {self.stats['skipped']}, "
            f"Errors: {self.stats['errors']}"
        )


class DataValidator:
    """Validates imported data before inserting."""

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate_dtc_code(self, code_data: Dict[str, Any]) -> bool:
        """Validate a DTC code record."""
        code = code_data.get("code", "")

        # Required fields
        if not code:
            self.errors.append("Missing required field: code")
            return False

        # Code format validation
        if len(code) != 5:
            self.errors.append(f"Invalid code length: {code}")
            return False

        if code[0] not in "PBCU":
            self.errors.append(f"Invalid code prefix: {code}")
            return False

        # Description validation
        if not code_data.get("description_en"):
            self.warnings.append(f"Missing English description for {code}")

        # Category validation
        category = code_data.get("category", "")
        valid_categories = ["powertrain", "body", "chassis", "network", "unknown"]
        if category and category not in valid_categories:
            self.warnings.append(f"Unknown category '{category}' for {code}")

        return True

    def validate_json_structure(self, data: Dict[str, Any]) -> bool:
        """Validate the structure of an export JSON file."""
        if not isinstance(data, dict):
            self.errors.append("Root element must be a dictionary")
            return False

        # Check for expected sections
        expected_sections = ["postgresql", "neo4j"]
        for section in expected_sections:
            if section not in data:
                self.warnings.append(f"Missing section: {section}")

        return True

    def get_report(self) -> str:
        """Get validation report."""
        report = []
        if self.errors:
            report.append("ERRORS:")
            for error in self.errors:
                report.append(f"  - {error}")
        if self.warnings:
            report.append("WARNINGS:")
            for warning in self.warnings:
                report.append(f"  - {warning}")
        if not self.errors and not self.warnings:
            report.append("Validation passed with no issues.")
        return "\n".join(report)


def get_sync_db_url() -> str:
    """Convert async database URL to sync."""
    url = settings.DATABASE_URL
    if url.startswith("postgresql+asyncpg://"):
        url = url.replace("postgresql+asyncpg://", "postgresql://")
    return url


def import_from_json(
    file_path: Path,
    resolver: ConflictResolver,
    dry_run: bool = False,
    validate_only: bool = False,
) -> bool:
    """
    Import data from JSON export file.

    Handles the comprehensive JSON export format from export_data.py.
    """
    logger.info(f"Importing from JSON: {file_path}")

    try:
        # Handle gzipped files
        if file_path.suffix == ".gz":
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                data = json.load(f)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

        # Validate structure
        validator = DataValidator()
        if not validator.validate_json_structure(data):
            logger.error("JSON structure validation failed")
            logger.error(validator.get_report())
            return False

        if validate_only:
            logger.info("Validation mode - checking data without importing")

        # Import PostgreSQL data
        pg_data = data.get("postgresql", {})
        if pg_data:
            logger.info("  Importing PostgreSQL data...")
            import_postgresql_data(pg_data, resolver, dry_run, validate_only)

        # Import Neo4j data
        neo4j_data = data.get("neo4j", {})
        if neo4j_data:
            logger.info("  Importing Neo4j data...")
            import_neo4j_data(neo4j_data, resolver, dry_run, validate_only)

        # Import Qdrant data
        qdrant_data = data.get("qdrant", {})
        if qdrant_data:
            logger.info("  Importing Qdrant data...")
            import_qdrant_data(qdrant_data, resolver, dry_run, validate_only)

        # Import translations
        translations = data.get("translations", {})
        if translations:
            logger.info("  Importing translations...")
            import_translations(translations, resolver, dry_run, validate_only)

        logger.info(f"JSON import complete. {resolver.get_summary()}")
        return True

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return False
    except Exception as e:
        logger.error(f"JSON import failed: {e}")
        return False


def import_postgresql_data(
    pg_data: Dict[str, Any],
    resolver: ConflictResolver,
    dry_run: bool = False,
    validate_only: bool = False,
) -> None:
    """Import PostgreSQL tables from export data."""
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import Session

    if validate_only:
        # Just validate data structure
        validator = DataValidator()
        dtc_codes = pg_data.get("dtc_codes", {}).get("data", [])
        for code_data in dtc_codes:
            validator.validate_dtc_code(code_data)
        logger.info(validator.get_report())
        return

    engine = create_engine(get_sync_db_url())

    with Session(engine) as session:
        for table_name, table_info in pg_data.items():
            if table_name in ("users", "alembic_version"):
                logger.info(f"    Skipping table: {table_name}")
                continue

            data_rows = table_info.get("data", [])
            columns = table_info.get("columns", [])

            if not data_rows:
                continue

            logger.info(f"    Processing table: {table_name} ({len(data_rows)} rows)")

            # Determine key field
            key_field = "code" if table_name == "dtc_codes" else "id"

            for row in data_rows:
                try:
                    # Check for existing record
                    existing = None
                    if key_field in row:
                        result = session.execute(
                            text(f'SELECT * FROM "{table_name}" WHERE "{key_field}" = :key'),
                            {"key": row[key_field]}
                        )
                        existing_row = result.fetchone()
                        if existing_row:
                            existing = dict(zip(columns, existing_row))

                    action, resolved_data = resolver.resolve(existing, row, key_field)

                    if dry_run:
                        logger.debug(f"      [DRY RUN] {action}: {row.get(key_field)}")
                        continue

                    if action == "insert":
                        # Build INSERT statement
                        cols = [f'"{c}"' for c in resolved_data.keys()]
                        placeholders = [f":{c.replace('.', '_')}" for c in resolved_data.keys()]
                        sql = f'INSERT INTO "{table_name}" ({", ".join(cols)}) VALUES ({", ".join(placeholders)})'

                        # Handle special types
                        params = {}
                        for k, v in resolved_data.items():
                            param_name = k.replace('.', '_')
                            if isinstance(v, (list, dict)):
                                params[param_name] = json.dumps(v)
                            else:
                                params[param_name] = v

                        session.execute(text(sql), params)

                    elif action == "update":
                        # Build UPDATE statement
                        set_clauses = [f'"{k}" = :{k.replace(".", "_")}' for k in resolved_data.keys() if k != key_field]
                        sql = f'UPDATE "{table_name}" SET {", ".join(set_clauses)} WHERE "{key_field}" = :_key'

                        params = {"_key": row[key_field]}
                        for k, v in resolved_data.items():
                            param_name = k.replace('.', '_')
                            if isinstance(v, (list, dict)):
                                params[param_name] = json.dumps(v)
                            else:
                                params[param_name] = v

                        session.execute(text(sql), params)

                except Exception as e:
                    logger.warning(f"      Error importing row: {e}")
                    resolver.stats["errors"] += 1

            session.commit()


def import_neo4j_data(
    neo4j_data: Dict[str, Any],
    resolver: ConflictResolver,
    dry_run: bool = False,
    validate_only: bool = False,
) -> None:
    """Import Neo4j nodes and relationships from export data."""
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )

        nodes = neo4j_data.get("nodes", {})
        relationships = neo4j_data.get("relationships", [])

        if validate_only:
            logger.info(f"    Would import {sum(len(n) for n in nodes.values())} nodes")
            logger.info(f"    Would import {len(relationships)} relationships")
            driver.close()
            return

        with driver.session() as session:
            # Import nodes
            for label, node_list in nodes.items():
                logger.info(f"    Importing {len(node_list)} {label} nodes...")

                for node in node_list:
                    # Remove internal fields
                    node_props = {k: v for k, v in node.items() if not k.startswith("_")}

                    # Determine unique key
                    if label == "DTCNode":
                        key_field = "code"
                    else:
                        key_field = "uid" if "uid" in node_props else "name"

                    key_value = node_props.get(key_field)
                    if not key_value:
                        continue

                    if dry_run:
                        logger.debug(f"      [DRY RUN] Would import {label}: {key_value}")
                        continue

                    # Check if exists
                    result = session.run(
                        f"MATCH (n:{label} {{{key_field}: $key}}) RETURN n",
                        key=key_value
                    )
                    existing = result.single()

                    if existing:
                        if resolver.strategy == "skip":
                            resolver.stats["skipped"] += 1
                            continue
                        elif resolver.strategy in ("overwrite", "merge", "newest"):
                            # Update existing
                            set_clause = ", ".join([f"n.{k} = ${k}" for k in node_props.keys()])
                            session.run(
                                f"MATCH (n:{label} {{{key_field}: $key}}) SET {set_clause}",
                                key=key_value,
                                **node_props
                            )
                            resolver.stats["updated"] += 1
                    else:
                        # Create new
                        props_clause = ", ".join([f"{k}: ${k}" for k in node_props.keys()])
                        session.run(
                            f"CREATE (n:{label} {{{props_clause}}})",
                            **node_props
                        )
                        resolver.stats["inserted"] += 1

            # Import relationships
            if relationships:
                logger.info(f"    Importing {len(relationships)} relationships...")

                for rel in relationships:
                    if dry_run:
                        logger.debug(f"      [DRY RUN] Would import relationship: {rel['type']}")
                        continue

                    # Get node identifiers
                    start_props = rel.get("start_props", {})
                    end_props = rel.get("end_props", {})

                    if not start_props or not end_props:
                        continue

                    # Use first property as identifier
                    start_key = list(start_props.keys())[0]
                    start_value = start_props[start_key]
                    end_key = list(end_props.keys())[0]
                    end_value = end_props[end_key]

                    rel_props = rel.get("properties", {})
                    props_clause = ", ".join([f"{k}: ${k}" for k in rel_props.keys()]) if rel_props else ""

                    query = f"""
                        MATCH (a:{rel['start_label']} {{{start_key}: $start_val}})
                        MATCH (b:{rel['end_label']} {{{end_key}: $end_val}})
                        MERGE (a)-[r:{rel['type']}]->(b)
                        {f'SET r = {{{props_clause}}}' if props_clause else ''}
                    """

                    try:
                        session.run(
                            query,
                            start_val=start_value,
                            end_val=end_value,
                            **rel_props
                        )
                    except Exception as e:
                        logger.warning(f"      Error creating relationship: {e}")
                        resolver.stats["errors"] += 1

        driver.close()

    except Exception as e:
        logger.warning(f"  Neo4j import failed: {e}")


def import_qdrant_data(
    qdrant_data: Dict[str, Any],
    resolver: ConflictResolver,
    dry_run: bool = False,
    validate_only: bool = False,
) -> None:
    """Import Qdrant collections from export data."""
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import Distance, PointStruct, VectorParams

        if settings.QDRANT_URL:
            client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
            )
        else:
            client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
            )

        collections = qdrant_data.get("collections", {})

        for name, collection_data in collections.items():
            points = collection_data.get("points", [])
            vector_size = collection_data.get("vector_size")
            distance = collection_data.get("distance", "Cosine")

            if validate_only:
                logger.info(f"    Would import collection {name}: {len(points)} points")
                continue

            logger.info(f"    Importing collection {name}: {len(points)} points...")

            if dry_run:
                logger.debug(f"      [DRY RUN] Would import {len(points)} points to {name}")
                continue

            # Check if collection exists
            existing_collections = [c.name for c in client.get_collections().collections]

            if name not in existing_collections and vector_size:
                # Create collection
                distance_type = Distance.COSINE
                if "EUCL" in str(distance).upper():
                    distance_type = Distance.EUCLID
                elif "DOT" in str(distance).upper():
                    distance_type = Distance.DOT

                client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(size=vector_size, distance=distance_type),
                )

            # Import points in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]

                point_structs = []
                for point in batch:
                    point_id = point.get("id")
                    payload = point.get("payload", {})
                    vector = point.get("vector")

                    if vector:
                        point_structs.append(PointStruct(
                            id=point_id,
                            vector=vector,
                            payload=payload,
                        ))

                if point_structs:
                    client.upsert(
                        collection_name=name,
                        points=point_structs,
                    )
                    resolver.stats["inserted"] += len(point_structs)

    except Exception as e:
        logger.warning(f"  Qdrant import failed: {e}")


def import_translations(
    translations: Dict[str, Any],
    resolver: ConflictResolver,
    dry_run: bool = False,
    validate_only: bool = False,
) -> None:
    """Import translation cache."""
    if validate_only:
        logger.info(f"    Would import {len(translations)} translations")
        return

    if dry_run:
        logger.debug(f"    [DRY RUN] Would import {len(translations)} translations")
        return

    cache_path = DATA_DIR / "dtc_codes" / "translation_cache.json"

    # Load existing cache
    existing_cache = {}
    if cache_path.exists():
        with open(cache_path, 'r', encoding='utf-8') as f:
            existing_cache = json.load(f)

    # Merge translations
    for key, value in translations.items():
        if key in existing_cache and resolver.strategy == "skip":
            resolver.stats["skipped"] += 1
        elif key in existing_cache:
            existing_cache[key] = value
            resolver.stats["updated"] += 1
        else:
            existing_cache[key] = value
            resolver.stats["inserted"] += 1

    # Save merged cache
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(existing_cache, f, indent=2, ensure_ascii=False)


def import_from_csv(
    directory_path: Path,
    resolver: ConflictResolver,
    dry_run: bool = False,
    validate_only: bool = False,
) -> bool:
    """
    Import data from CSV directory.

    Expects CSV files following the naming convention from export_data.py:
    - pg_<table_name>.csv for PostgreSQL tables
    - neo4j_<label>.csv for Neo4j nodes
    - neo4j_relationships.csv for relationships
    """
    logger.info(f"Importing from CSV directory: {directory_path}")

    try:
        if not directory_path.is_dir():
            logger.error(f"Not a directory: {directory_path}")
            return False

        csv_files = list(directory_path.glob("*.csv"))
        if not csv_files:
            logger.error(f"No CSV files found in {directory_path}")
            return False

        logger.info(f"  Found {len(csv_files)} CSV files")

        for csv_file in csv_files:
            logger.info(f"  Processing: {csv_file.name}")

            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            if csv_file.name.startswith("pg_"):
                table_name = csv_file.name[3:-4]  # Remove 'pg_' prefix and '.csv' suffix
                import_postgresql_csv(table_name, rows, resolver, dry_run, validate_only)

            elif csv_file.name.startswith("neo4j_") and csv_file.name != "neo4j_relationships.csv":
                label = csv_file.name[6:-4]  # Remove 'neo4j_' prefix and '.csv' suffix
                import_neo4j_csv(label, rows, resolver, dry_run, validate_only)

            elif csv_file.name == "neo4j_relationships.csv":
                import_neo4j_relationships_csv(rows, resolver, dry_run, validate_only)

            elif csv_file.name == "dtc_codes_full.csv":
                import_dtc_codes_csv(rows, resolver, dry_run, validate_only)

        logger.info(f"CSV import complete. {resolver.get_summary()}")
        return True

    except Exception as e:
        logger.error(f"CSV import failed: {e}")
        return False


def import_postgresql_csv(
    table_name: str,
    rows: List[Dict[str, Any]],
    resolver: ConflictResolver,
    dry_run: bool = False,
    validate_only: bool = False,
) -> None:
    """Import CSV data to PostgreSQL table."""
    if table_name in ("users", "alembic_version"):
        logger.info(f"    Skipping table: {table_name}")
        return

    if validate_only:
        logger.info(f"    Would import {len(rows)} rows to {table_name}")
        return

    if dry_run:
        logger.debug(f"    [DRY RUN] Would import {len(rows)} rows to {table_name}")
        return

    # Parse JSON strings back to Python objects
    for row in rows:
        for key, value in row.items():
            if isinstance(value, str) and value.startswith(('[', '{')):
                try:
                    row[key] = json.loads(value)
                except json.JSONDecodeError:
                    pass

    # Use the PostgreSQL import function
    import_postgresql_data({table_name: {"data": rows, "columns": list(rows[0].keys())}}, resolver, dry_run, validate_only)


def import_neo4j_csv(
    label: str,
    rows: List[Dict[str, Any]],
    resolver: ConflictResolver,
    dry_run: bool = False,
    validate_only: bool = False,
) -> None:
    """Import CSV data to Neo4j as nodes."""
    if validate_only:
        logger.info(f"    Would import {len(rows)} {label} nodes")
        return

    # Parse JSON strings
    for row in rows:
        for key, value in row.items():
            if isinstance(value, str) and value.startswith(('[', '{')):
                try:
                    row[key] = json.loads(value)
                except json.JSONDecodeError:
                    pass

    import_neo4j_data({"nodes": {label: rows}, "relationships": []}, resolver, dry_run, validate_only)


def import_neo4j_relationships_csv(
    rows: List[Dict[str, Any]],
    resolver: ConflictResolver,
    dry_run: bool = False,
    validate_only: bool = False,
) -> None:
    """Import CSV data as Neo4j relationships."""
    if validate_only:
        logger.info(f"    Would import {len(rows)} relationships")
        return

    # Convert CSV rows to relationship format
    relationships = []
    for row in rows:
        rel_props = {}
        if row.get("rel_props"):
            try:
                rel_props = json.loads(row["rel_props"])
            except json.JSONDecodeError:
                pass

        relationships.append({
            "start_label": row.get("start_label"),
            "start_id": row.get("start_id"),
            "type": row.get("rel_type"),
            "properties": rel_props,
            "end_label": row.get("end_label"),
            "end_id": row.get("end_id"),
        })

    import_neo4j_data({"nodes": {}, "relationships": relationships}, resolver, dry_run, validate_only)


def import_dtc_codes_csv(
    rows: List[Dict[str, Any]],
    resolver: ConflictResolver,
    dry_run: bool = False,
    validate_only: bool = False,
) -> None:
    """Import DTC codes from CSV to the merged JSON file."""
    if validate_only:
        logger.info(f"    Would import {len(rows)} DTC codes")
        return

    if dry_run:
        logger.debug(f"    [DRY RUN] Would import {len(rows)} DTC codes")
        return

    # Parse JSON strings
    for row in rows:
        for key, value in row.items():
            if isinstance(value, str) and value.startswith(('[', '{')):
                try:
                    row[key] = json.loads(value)
                except json.JSONDecodeError:
                    pass

    # Load existing merged codes
    merged_path = DATA_DIR / "dtc_codes" / "all_codes_merged.json"
    existing_data = {"metadata": {}, "codes": []}

    if merged_path.exists():
        with open(merged_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)

    existing_codes = {c["code"]: c for c in existing_data.get("codes", [])}

    # Import codes
    for row in rows:
        code = row.get("code")
        if not code:
            continue

        existing = existing_codes.get(code)
        action, resolved = resolver.resolve(existing, row, "code")

        if action in ("insert", "update"):
            existing_codes[code] = resolved

    # Save updated data
    existing_data["codes"] = list(existing_codes.values())
    existing_data["metadata"]["last_import"] = datetime.now().isoformat()

    with open(merged_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False)


def import_from_sqlite(
    file_path: Path,
    resolver: ConflictResolver,
    dry_run: bool = False,
    validate_only: bool = False,
) -> bool:
    """Import data from SQLite database."""
    logger.info(f"Importing from SQLite: {file_path}")

    try:
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return False

        conn = sqlite3.connect(file_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        logger.info(f"  Found {len(tables)} tables")

        for table_name in tables:
            if table_name.startswith("_"):  # Skip metadata tables
                continue

            cursor.execute(f'SELECT * FROM "{table_name}"')
            rows = [dict(row) for row in cursor.fetchall()]

            if not rows:
                continue

            logger.info(f"  Processing table: {table_name} ({len(rows)} rows)")

            # Parse JSON strings
            for row in rows:
                for key, value in row.items():
                    if isinstance(value, str) and value.startswith(('[', '{')):
                        try:
                            row[key] = json.loads(value)
                        except json.JSONDecodeError:
                            pass

            if table_name.startswith("neo4j_"):
                if table_name == "neo4j_relationships":
                    import_neo4j_relationships_csv(rows, resolver, dry_run, validate_only)
                else:
                    label = table_name[6:]  # Remove 'neo4j_' prefix
                    import_neo4j_csv(label, rows, resolver, dry_run, validate_only)
            else:
                import_postgresql_csv(table_name, rows, resolver, dry_run, validate_only)

        conn.close()
        logger.info(f"SQLite import complete. {resolver.get_summary()}")
        return True

    except Exception as e:
        logger.error(f"SQLite import failed: {e}")
        return False


def import_from_graphml(
    file_path: Path,
    resolver: ConflictResolver,
    dry_run: bool = False,
    validate_only: bool = False,
) -> bool:
    """Import Neo4j graph from GraphML format."""
    logger.info(f"Importing from GraphML: {file_path}")

    try:
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return False

        tree = ET.parse(file_path)
        root = tree.getroot()

        # Handle namespace
        ns = {"graphml": "http://graphml.graphdrawing.org/xmlns"}
        if root.tag.startswith("{"):
            ns_uri = root.tag[1:root.tag.index("}")]
            ns = {"graphml": ns_uri}

        # Parse key definitions
        keys = {}
        for key_elem in root.findall("graphml:key", ns) or root.findall("key"):
            key_id = key_elem.get("id")
            attr_name = key_elem.get("attr.name")
            for_type = key_elem.get("for")
            keys[key_id] = {"name": attr_name, "for": for_type}

        # Find graph element
        graph = root.find("graphml:graph", ns) or root.find("graph")
        if graph is None:
            logger.error("No graph element found in GraphML")
            return False

        # Parse nodes
        nodes_by_label: Dict[str, List[Dict[str, Any]]] = {}
        node_id_map: Dict[str, str] = {}  # graph_id -> element_id

        for node_elem in graph.findall("graphml:node", ns) or graph.findall("node"):
            node_id = node_elem.get("id")
            node_data: Dict[str, Any] = {}
            label = "Unknown"

            for data_elem in node_elem.findall("graphml:data", ns) or node_elem.findall("data"):
                key_id = data_elem.get("key")
                key_info = keys.get(key_id, {})
                attr_name = key_info.get("name", key_id)
                value = data_elem.text

                if attr_name == "label":
                    label = value
                else:
                    # Try to parse JSON values
                    if value and value.startswith(('[', '{')):
                        try:
                            value = json.loads(value)
                        except json.JSONDecodeError:
                            pass
                    node_data[attr_name] = value

            if label not in nodes_by_label:
                nodes_by_label[label] = []
            nodes_by_label[label].append(node_data)
            node_id_map[node_id] = node_data.get("code") or node_data.get("uid") or node_id

        logger.info(f"  Found {sum(len(nodes) for nodes in nodes_by_label.values())} nodes")

        # Parse edges
        relationships = []

        for edge_elem in graph.findall("graphml:edge", ns) or graph.findall("edge"):
            source_id = edge_elem.get("source")
            target_id = edge_elem.get("target")
            rel_type = "UNKNOWN"
            rel_props: Dict[str, Any] = {}

            for data_elem in edge_elem.findall("graphml:data", ns) or edge_elem.findall("data"):
                key_id = data_elem.get("key")
                key_info = keys.get(key_id, {})
                attr_name = key_info.get("name", key_id)
                value = data_elem.text

                if attr_name == "type":
                    rel_type = value
                else:
                    if value and value.startswith(('[', '{')):
                        try:
                            value = json.loads(value)
                        except json.JSONDecodeError:
                            pass
                    rel_props[attr_name] = value

            relationships.append({
                "start_id": source_id,
                "end_id": target_id,
                "type": rel_type,
                "properties": rel_props,
            })

        logger.info(f"  Found {len(relationships)} relationships")

        if validate_only:
            logger.info("  Validation complete")
            return True

        # Import to Neo4j
        import_neo4j_data(
            {"nodes": nodes_by_label, "relationships": relationships},
            resolver,
            dry_run,
            validate_only,
        )

        logger.info(f"GraphML import complete. {resolver.get_summary()}")
        return True

    except ET.ParseError as e:
        logger.error(f"XML parse error: {e}")
        return False
    except Exception as e:
        logger.error(f"GraphML import failed: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Import data to AutoCognitix from exported formats"
    )

    # Source options
    source_group = parser.add_argument_group("Import Sources")
    source_group.add_argument(
        "--json",
        type=str,
        metavar="FILE",
        help="Import from JSON export file",
    )
    source_group.add_argument(
        "--csv",
        type=str,
        metavar="DIR",
        help="Import from CSV directory",
    )
    source_group.add_argument(
        "--sqlite",
        type=str,
        metavar="FILE",
        help="Import from SQLite database",
    )
    source_group.add_argument(
        "--graphml",
        type=str,
        metavar="FILE",
        help="Import from GraphML file",
    )

    # Conflict resolution
    parser.add_argument(
        "--on-conflict",
        type=str,
        choices=ConflictResolver.STRATEGIES,
        default="skip",
        help="Conflict resolution strategy (default: skip)",
    )

    # Other options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview import without making changes",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate data only, do not import",
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

    # Check that at least one source is specified
    if not any([args.json, args.csv, args.sqlite, args.graphml]):
        parser.error("At least one import source must be specified")

    # Create conflict resolver
    resolver = ConflictResolver(args.on_conflict)

    logger.info(f"Conflict resolution strategy: {args.on_conflict}")
    if args.dry_run:
        logger.info("DRY RUN MODE - No changes will be made")
    if args.validate:
        logger.info("VALIDATION MODE - Data will be validated only")

    results = []

    try:
        if args.json:
            result = import_from_json(
                Path(args.json),
                resolver,
                args.dry_run,
                args.validate,
            )
            results.append(("JSON", result))

        if args.csv:
            result = import_from_csv(
                Path(args.csv),
                resolver,
                args.dry_run,
                args.validate,
            )
            results.append(("CSV", result))

        if args.sqlite:
            result = import_from_sqlite(
                Path(args.sqlite),
                resolver,
                args.dry_run,
                args.validate,
            )
            results.append(("SQLite", result))

        if args.graphml:
            result = import_from_graphml(
                Path(args.graphml),
                resolver,
                args.dry_run,
                args.validate,
            )
            results.append(("GraphML", result))

        # Summary
        logger.info("=" * 50)
        logger.info("IMPORT SUMMARY")
        logger.info("=" * 50)
        for name, success in results:
            status = "SUCCESS" if success else "FAILED"
            logger.info(f"  {name}: {status}")
        logger.info(f"Statistics: {resolver.get_summary()}")
        logger.info("=" * 50)

        # Exit with error if any import failed
        if not all(success for _, success in results):
            sys.exit(1)

    except Exception as e:
        logger.error(f"Import failed: {e}")
        raise


if __name__ == "__main__":
    main()
