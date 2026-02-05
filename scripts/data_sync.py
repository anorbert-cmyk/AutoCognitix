#!/usr/bin/env python3
"""
Data Synchronization Script for AutoCognitix

Synchronizes data between PostgreSQL, Neo4j, and Qdrant databases:
- PostgreSQL <-> Neo4j: Ensures DTC codes and relationships are consistent
- PostgreSQL <-> Qdrant: Ensures vector embeddings are up-to-date
- Translation sync: Synchronizes translations across all databases
- Consistency verification: Reports discrepancies between databases

Usage:
    # Sync all databases
    python scripts/data_sync.py --all

    # Sync specific pairs
    python scripts/data_sync.py --postgres-neo4j          # Sync PostgreSQL to Neo4j
    python scripts/data_sync.py --postgres-qdrant         # Sync PostgreSQL to Qdrant
    python scripts/data_sync.py --neo4j-postgres          # Sync Neo4j to PostgreSQL
    python scripts/data_sync.py --translations            # Sync translations

    # Verification only
    python scripts/data_sync.py --verify                  # Check consistency without syncing
    python scripts/data_sync.py --verify --report FILE    # Save verification report to file

    # Options
    python scripts/data_sync.py --all --dry-run           # Preview changes without applying
    python scripts/data_sync.py --all --force             # Force sync even with conflicts
    python scripts/data_sync.py --all --batch-size 100    # Batch size for large syncs
"""

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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


@dataclass
class SyncStats:
    """Statistics for a sync operation."""
    source: str
    target: str
    records_checked: int = 0
    records_added: int = 0
    records_updated: int = 0
    records_skipped: int = 0
    records_deleted: int = 0
    errors: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    def complete(self) -> None:
        """Mark sync as complete."""
        self.end_time = datetime.now()

    @property
    def duration_seconds(self) -> float:
        """Get sync duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source": self.source,
            "target": self.target,
            "records_checked": self.records_checked,
            "records_added": self.records_added,
            "records_updated": self.records_updated,
            "records_skipped": self.records_skipped,
            "records_deleted": self.records_deleted,
            "errors": self.errors,
            "duration_seconds": self.duration_seconds,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
        }


@dataclass
class VerificationReport:
    """Report of data consistency verification."""
    timestamp: datetime = field(default_factory=datetime.now)
    postgres_count: int = 0
    neo4j_count: int = 0
    qdrant_count: int = 0
    missing_in_neo4j: List[str] = field(default_factory=list)
    missing_in_postgres: List[str] = field(default_factory=list)
    missing_in_qdrant: List[str] = field(default_factory=list)
    extra_in_neo4j: List[str] = field(default_factory=list)
    extra_in_qdrant: List[str] = field(default_factory=list)
    content_mismatches: List[Dict[str, Any]] = field(default_factory=list)
    translation_mismatches: List[Dict[str, Any]] = field(default_factory=list)
    is_consistent: bool = True

    def add_issue(self, issue_type: str, details: Any) -> None:
        """Add an issue to the report."""
        self.is_consistent = False
        if issue_type == "missing_in_neo4j":
            self.missing_in_neo4j.append(details)
        elif issue_type == "missing_in_postgres":
            self.missing_in_postgres.append(details)
        elif issue_type == "missing_in_qdrant":
            self.missing_in_qdrant.append(details)
        elif issue_type == "extra_in_neo4j":
            self.extra_in_neo4j.append(details)
        elif issue_type == "extra_in_qdrant":
            self.extra_in_qdrant.append(details)
        elif issue_type == "content_mismatch":
            self.content_mismatches.append(details)
        elif issue_type == "translation_mismatch":
            self.translation_mismatches.append(details)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "is_consistent": self.is_consistent,
            "counts": {
                "postgres": self.postgres_count,
                "neo4j": self.neo4j_count,
                "qdrant": self.qdrant_count,
            },
            "issues": {
                "missing_in_neo4j": self.missing_in_neo4j[:50],  # Limit for readability
                "missing_in_postgres": self.missing_in_postgres[:50],
                "missing_in_qdrant": self.missing_in_qdrant[:50],
                "extra_in_neo4j": self.extra_in_neo4j[:50],
                "extra_in_qdrant": self.extra_in_qdrant[:50],
                "content_mismatches": self.content_mismatches[:50],
                "translation_mismatches": self.translation_mismatches[:50],
            },
            "issue_counts": {
                "missing_in_neo4j": len(self.missing_in_neo4j),
                "missing_in_postgres": len(self.missing_in_postgres),
                "missing_in_qdrant": len(self.missing_in_qdrant),
                "extra_in_neo4j": len(self.extra_in_neo4j),
                "extra_in_qdrant": len(self.extra_in_qdrant),
                "content_mismatches": len(self.content_mismatches),
                "translation_mismatches": len(self.translation_mismatches),
            },
        }

    def print_summary(self) -> None:
        """Print a summary of the report."""
        logger.info("=" * 60)
        logger.info("VERIFICATION REPORT")
        logger.info("=" * 60)
        logger.info(f"Status: {'CONSISTENT' if self.is_consistent else 'INCONSISTENT'}")
        logger.info("")
        logger.info("Record Counts:")
        logger.info(f"  PostgreSQL: {self.postgres_count}")
        logger.info(f"  Neo4j:      {self.neo4j_count}")
        logger.info(f"  Qdrant:     {self.qdrant_count}")
        logger.info("")

        if not self.is_consistent:
            logger.info("Issues Found:")
            if self.missing_in_neo4j:
                logger.info(f"  Missing in Neo4j:       {len(self.missing_in_neo4j)}")
            if self.missing_in_postgres:
                logger.info(f"  Missing in PostgreSQL:  {len(self.missing_in_postgres)}")
            if self.missing_in_qdrant:
                logger.info(f"  Missing in Qdrant:      {len(self.missing_in_qdrant)}")
            if self.extra_in_neo4j:
                logger.info(f"  Extra in Neo4j:         {len(self.extra_in_neo4j)}")
            if self.extra_in_qdrant:
                logger.info(f"  Extra in Qdrant:        {len(self.extra_in_qdrant)}")
            if self.content_mismatches:
                logger.info(f"  Content mismatches:     {len(self.content_mismatches)}")
            if self.translation_mismatches:
                logger.info(f"  Translation mismatches: {len(self.translation_mismatches)}")

        logger.info("=" * 60)


def get_sync_db_url() -> str:
    """Convert async database URL to sync."""
    url = settings.DATABASE_URL
    if url.startswith("postgresql+asyncpg://"):
        url = url.replace("postgresql+asyncpg://", "postgresql://")
    return url


def get_postgres_dtc_codes() -> Dict[str, Dict[str, Any]]:
    """Get all DTC codes from PostgreSQL."""
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import Session

    engine = create_engine(get_sync_db_url())
    codes: Dict[str, Dict[str, Any]] = {}

    with Session(engine) as session:
        try:
            result = session.execute(text("SELECT * FROM dtc_codes"))
            columns = result.keys()
            for row in result.fetchall():
                row_dict = dict(zip(columns, row))
                code = row_dict.get("code")
                if code:
                    codes[code] = row_dict
        except Exception as e:
            logger.warning(f"Could not query PostgreSQL dtc_codes: {e}")

    return codes


def get_neo4j_dtc_codes() -> Dict[str, Dict[str, Any]]:
    """Get all DTC codes from Neo4j."""
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )

        codes: Dict[str, Dict[str, Any]] = {}

        with driver.session() as session:
            result = session.run("MATCH (n:DTCNode) RETURN n")
            for record in result:
                node = record["n"]
                node_dict = dict(node)
                code = node_dict.get("code")
                if code:
                    codes[code] = node_dict

        driver.close()
        return codes

    except Exception as e:
        logger.warning(f"Could not query Neo4j: {e}")
        return {}


def get_qdrant_dtc_codes() -> Dict[str, Dict[str, Any]]:
    """Get all DTC codes from Qdrant."""
    try:
        from qdrant_client import QdrantClient

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

        codes: Dict[str, Dict[str, Any]] = {}

        # Check if collection exists
        try:
            collections = [c.name for c in client.get_collections().collections]
            collection_name = "dtc_codes"  # Default collection name

            if collection_name not in collections:
                logger.warning(f"Qdrant collection '{collection_name}' not found")
                return codes

            # Scroll through all points
            offset = None
            while True:
                result = client.scroll(
                    collection_name=collection_name,
                    offset=offset,
                    limit=100,
                    with_payload=True,
                    with_vectors=False,
                )

                points, next_offset = result

                for point in points:
                    payload = point.payload or {}
                    code = payload.get("code")
                    if code:
                        codes[code] = payload

                if next_offset is None:
                    break
                offset = next_offset

        except Exception as e:
            logger.warning(f"Could not scroll Qdrant collection: {e}")

        return codes

    except Exception as e:
        logger.warning(f"Could not connect to Qdrant: {e}")
        return {}


def compute_content_hash(data: Dict[str, Any], fields: List[str]) -> str:
    """Compute a hash of specified fields for comparison."""
    content_parts = []
    for field in sorted(fields):
        value = data.get(field, "")
        if isinstance(value, (list, dict)):
            value = json.dumps(value, sort_keys=True, ensure_ascii=False)
        content_parts.append(f"{field}:{value}")

    content_str = "|".join(content_parts)
    return hashlib.md5(content_str.encode('utf-8')).hexdigest()


def verify_consistency(report_path: Optional[Path] = None) -> VerificationReport:
    """
    Verify data consistency across all databases.

    Returns:
        VerificationReport with details of any inconsistencies.
    """
    logger.info("Starting data consistency verification...")

    report = VerificationReport()

    # Get data from all sources
    logger.info("  Fetching data from PostgreSQL...")
    pg_codes = get_postgres_dtc_codes()
    report.postgres_count = len(pg_codes)

    logger.info("  Fetching data from Neo4j...")
    neo4j_codes = get_neo4j_dtc_codes()
    report.neo4j_count = len(neo4j_codes)

    logger.info("  Fetching data from Qdrant...")
    qdrant_codes = get_qdrant_dtc_codes()
    report.qdrant_count = len(qdrant_codes)

    # Compare PostgreSQL vs Neo4j
    logger.info("  Comparing PostgreSQL vs Neo4j...")
    pg_code_set = set(pg_codes.keys())
    neo4j_code_set = set(neo4j_codes.keys())

    missing_in_neo4j = pg_code_set - neo4j_code_set
    extra_in_neo4j = neo4j_code_set - pg_code_set

    for code in missing_in_neo4j:
        report.add_issue("missing_in_neo4j", code)

    for code in extra_in_neo4j:
        report.add_issue("extra_in_neo4j", code)

    # Compare content for codes present in both
    common_codes = pg_code_set & neo4j_code_set
    comparison_fields = ["description_en", "description_hu", "category", "severity"]

    for code in common_codes:
        pg_hash = compute_content_hash(pg_codes[code], comparison_fields)
        neo4j_hash = compute_content_hash(neo4j_codes[code], comparison_fields)

        if pg_hash != neo4j_hash:
            report.add_issue("content_mismatch", {
                "code": code,
                "source": "postgres_vs_neo4j",
                "postgres": {f: pg_codes[code].get(f) for f in comparison_fields},
                "neo4j": {f: neo4j_codes[code].get(f) for f in comparison_fields},
            })

    # Compare PostgreSQL vs Qdrant
    logger.info("  Comparing PostgreSQL vs Qdrant...")
    qdrant_code_set = set(qdrant_codes.keys())

    missing_in_qdrant = pg_code_set - qdrant_code_set
    extra_in_qdrant = qdrant_code_set - pg_code_set

    for code in missing_in_qdrant:
        report.add_issue("missing_in_qdrant", code)

    for code in extra_in_qdrant:
        report.add_issue("extra_in_qdrant", code)

    # Check translation consistency
    logger.info("  Checking translation consistency...")
    translation_cache_path = DATA_DIR / "dtc_codes" / "translation_cache.json"

    if translation_cache_path.exists():
        with open(translation_cache_path, 'r', encoding='utf-8') as f:
            translations = json.load(f)

        for code, pg_data in pg_codes.items():
            pg_hu = pg_data.get("description_hu", "")
            neo4j_hu = neo4j_codes.get(code, {}).get("description_hu", "")

            if pg_hu and neo4j_hu and pg_hu != neo4j_hu:
                report.add_issue("translation_mismatch", {
                    "code": code,
                    "postgres": pg_hu,
                    "neo4j": neo4j_hu,
                })

    # Print summary
    report.print_summary()

    # Save report if path provided
    if report_path:
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Report saved to: {report_path}")

    return report


def sync_postgres_to_neo4j(
    dry_run: bool = False,
    force: bool = False,
    batch_size: int = 100,
) -> SyncStats:
    """
    Sync DTC codes from PostgreSQL to Neo4j.

    Creates/updates DTCNode nodes and their relationships.
    """
    stats = SyncStats(source="PostgreSQL", target="Neo4j")
    logger.info("Syncing PostgreSQL -> Neo4j...")

    try:
        from neo4j import GraphDatabase

        # Get data from both sources
        pg_codes = get_postgres_dtc_codes()
        neo4j_codes = get_neo4j_dtc_codes()

        driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )

        with driver.session() as session:
            for code, pg_data in pg_codes.items():
                stats.records_checked += 1

                existing = neo4j_codes.get(code)

                if existing is None:
                    # Create new node
                    if dry_run:
                        logger.debug(f"  [DRY RUN] Would create: {code}")
                    else:
                        # Prepare properties
                        props = {
                            "code": code,
                            "description_en": pg_data.get("description_en", ""),
                            "description_hu": pg_data.get("description_hu", ""),
                            "category": pg_data.get("category", ""),
                            "severity": pg_data.get("severity", ""),
                            "synced_at": datetime.now().isoformat(),
                        }

                        # Remove None values
                        props = {k: v for k, v in props.items() if v is not None}

                        props_clause = ", ".join([f"{k}: ${k}" for k in props.keys()])
                        session.run(
                            f"CREATE (n:DTCNode {{{props_clause}}})",
                            **props
                        )
                        logger.debug(f"  Created: {code}")
                    stats.records_added += 1

                elif force:
                    # Update existing node
                    if dry_run:
                        logger.debug(f"  [DRY RUN] Would update: {code}")
                    else:
                        props = {
                            "description_en": pg_data.get("description_en", ""),
                            "description_hu": pg_data.get("description_hu", ""),
                            "category": pg_data.get("category", ""),
                            "severity": pg_data.get("severity", ""),
                            "synced_at": datetime.now().isoformat(),
                        }
                        props = {k: v for k, v in props.items() if v is not None}

                        set_clause = ", ".join([f"n.{k} = ${k}" for k in props.keys()])
                        session.run(
                            f"MATCH (n:DTCNode {{code: $code}}) SET {set_clause}",
                            code=code,
                            **props
                        )
                        logger.debug(f"  Updated: {code}")
                    stats.records_updated += 1
                else:
                    stats.records_skipped += 1

        driver.close()

    except Exception as e:
        logger.error(f"PostgreSQL -> Neo4j sync failed: {e}")
        stats.errors += 1

    stats.complete()
    return stats


def sync_postgres_to_qdrant(
    dry_run: bool = False,
    force: bool = False,
    batch_size: int = 100,
) -> SyncStats:
    """
    Sync DTC codes from PostgreSQL to Qdrant.

    Creates/updates vector embeddings for DTC codes.
    """
    stats = SyncStats(source="PostgreSQL", target="Qdrant")
    logger.info("Syncing PostgreSQL -> Qdrant...")

    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import Distance, PointStruct, VectorParams

        # Get data
        pg_codes = get_postgres_dtc_codes()
        qdrant_codes = get_qdrant_dtc_codes()

        # Initialize Qdrant client
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

        collection_name = "dtc_codes"

        # Check if collection exists, create if not
        existing_collections = [c.name for c in client.get_collections().collections]
        if collection_name not in existing_collections:
            logger.info(f"  Creating Qdrant collection: {collection_name}")
            if not dry_run:
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=settings.EMBEDDING_DIMENSION,  # 768 for huBERT
                        distance=Distance.COSINE,
                    ),
                )

        # Try to load embedding service for generating vectors
        embedding_service = None
        try:
            sys.path.insert(0, str(PROJECT_ROOT / "backend"))
            from app.services.embedding_service import EmbeddingService
            embedding_service = EmbeddingService()
            logger.info("  Embedding service loaded")
        except Exception as e:
            logger.warning(f"  Could not load embedding service: {e}")
            logger.warning("  Will sync payloads only, no new vectors")

        # Prepare points to upsert
        points_to_upsert: List[PointStruct] = []

        for code, pg_data in pg_codes.items():
            stats.records_checked += 1

            existing = qdrant_codes.get(code)

            if existing is None or force:
                # Prepare payload
                payload = {
                    "code": code,
                    "description_en": pg_data.get("description_en", ""),
                    "description_hu": pg_data.get("description_hu", ""),
                    "category": pg_data.get("category", ""),
                    "severity": pg_data.get("severity", ""),
                    "synced_at": datetime.now().isoformat(),
                }

                # Generate embedding if service available
                vector = None
                if embedding_service:
                    text = f"{code} {payload['description_en']} {payload['description_hu']}"
                    try:
                        vector = embedding_service.get_embedding(text)
                    except Exception as e:
                        logger.warning(f"    Could not generate embedding for {code}: {e}")

                if vector:
                    if dry_run:
                        logger.debug(f"  [DRY RUN] Would upsert: {code}")
                    else:
                        # Use code as point ID (converted to integer hash)
                        point_id = abs(hash(code)) % (2**63)
                        points_to_upsert.append(PointStruct(
                            id=point_id,
                            vector=vector,
                            payload=payload,
                        ))

                    if existing is None:
                        stats.records_added += 1
                    else:
                        stats.records_updated += 1
                else:
                    stats.records_skipped += 1
            else:
                stats.records_skipped += 1

            # Batch upsert
            if len(points_to_upsert) >= batch_size:
                if not dry_run:
                    client.upsert(
                        collection_name=collection_name,
                        points=points_to_upsert,
                    )
                points_to_upsert = []

        # Final batch
        if points_to_upsert and not dry_run:
            client.upsert(
                collection_name=collection_name,
                points=points_to_upsert,
            )

    except Exception as e:
        logger.error(f"PostgreSQL -> Qdrant sync failed: {e}")
        stats.errors += 1

    stats.complete()
    return stats


def sync_neo4j_to_postgres(
    dry_run: bool = False,
    force: bool = False,
    batch_size: int = 100,
) -> SyncStats:
    """
    Sync DTC codes from Neo4j to PostgreSQL.

    Updates PostgreSQL with any codes or changes from Neo4j.
    """
    stats = SyncStats(source="Neo4j", target="PostgreSQL")
    logger.info("Syncing Neo4j -> PostgreSQL...")

    try:
        from sqlalchemy import create_engine, text
        from sqlalchemy.orm import Session

        # Get data
        pg_codes = get_postgres_dtc_codes()
        neo4j_codes = get_neo4j_dtc_codes()

        engine = create_engine(get_sync_db_url())

        with Session(engine) as session:
            for code, neo4j_data in neo4j_codes.items():
                stats.records_checked += 1

                existing = pg_codes.get(code)

                if existing is None:
                    # Insert new record
                    if dry_run:
                        logger.debug(f"  [DRY RUN] Would insert: {code}")
                    else:
                        session.execute(
                            text("""
                                INSERT INTO dtc_codes (code, description_en, description_hu, category, severity)
                                VALUES (:code, :description_en, :description_hu, :category, :severity)
                            """),
                            {
                                "code": code,
                                "description_en": neo4j_data.get("description_en", ""),
                                "description_hu": neo4j_data.get("description_hu", ""),
                                "category": neo4j_data.get("category", ""),
                                "severity": neo4j_data.get("severity", ""),
                            }
                        )
                        logger.debug(f"  Inserted: {code}")
                    stats.records_added += 1

                elif force:
                    # Update existing record
                    if dry_run:
                        logger.debug(f"  [DRY RUN] Would update: {code}")
                    else:
                        session.execute(
                            text("""
                                UPDATE dtc_codes
                                SET description_en = :description_en,
                                    description_hu = :description_hu,
                                    category = :category,
                                    severity = :severity,
                                    updated_at = NOW()
                                WHERE code = :code
                            """),
                            {
                                "code": code,
                                "description_en": neo4j_data.get("description_en", ""),
                                "description_hu": neo4j_data.get("description_hu", ""),
                                "category": neo4j_data.get("category", ""),
                                "severity": neo4j_data.get("severity", ""),
                            }
                        )
                        logger.debug(f"  Updated: {code}")
                    stats.records_updated += 1
                else:
                    stats.records_skipped += 1

            if not dry_run:
                session.commit()

    except Exception as e:
        logger.error(f"Neo4j -> PostgreSQL sync failed: {e}")
        stats.errors += 1

    stats.complete()
    return stats


def sync_translations(
    dry_run: bool = False,
    force: bool = False,
) -> SyncStats:
    """
    Sync translations across all databases.

    Uses the translation cache as the source of truth for Hungarian translations.
    """
    stats = SyncStats(source="TranslationCache", target="All")
    logger.info("Syncing translations...")

    try:
        from sqlalchemy import create_engine, text
        from sqlalchemy.orm import Session
        from neo4j import GraphDatabase

        # Load translation cache
        translation_cache_path = DATA_DIR / "dtc_codes" / "translation_cache.json"

        if not translation_cache_path.exists():
            logger.warning("Translation cache not found")
            return stats

        with open(translation_cache_path, 'r', encoding='utf-8') as f:
            translation_cache = json.load(f)

        # Get current data
        pg_codes = get_postgres_dtc_codes()
        neo4j_codes = get_neo4j_dtc_codes()

        # Create hash -> translation mapping
        translations: Dict[str, str] = {}
        for key, value in translation_cache.items():
            if isinstance(value, dict):
                translations[key] = value.get("translation", "")
            else:
                translations[key] = value

        # Sync to PostgreSQL
        logger.info("  Syncing translations to PostgreSQL...")
        engine = create_engine(get_sync_db_url())

        with Session(engine) as session:
            for code, pg_data in pg_codes.items():
                desc_en = pg_data.get("description_en", "")
                current_hu = pg_data.get("description_hu", "")

                # Create hash from English description
                if desc_en:
                    desc_hash = hashlib.md5(desc_en.encode('utf-8')).hexdigest()
                    translation = translations.get(desc_hash)

                    if translation and (not current_hu or force):
                        stats.records_checked += 1

                        if dry_run:
                            logger.debug(f"  [DRY RUN] Would update translation: {code}")
                        else:
                            session.execute(
                                text("""
                                    UPDATE dtc_codes
                                    SET description_hu = :description_hu,
                                        updated_at = NOW()
                                    WHERE code = :code
                                """),
                                {"code": code, "description_hu": translation}
                            )
                        stats.records_updated += 1

            if not dry_run:
                session.commit()

        # Sync to Neo4j
        logger.info("  Syncing translations to Neo4j...")
        try:
            driver = GraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
            )

            with driver.session() as neo4j_session:
                for code, neo4j_data in neo4j_codes.items():
                    desc_en = neo4j_data.get("description_en", "")
                    current_hu = neo4j_data.get("description_hu", "")

                    if desc_en:
                        desc_hash = hashlib.md5(desc_en.encode('utf-8')).hexdigest()
                        translation = translations.get(desc_hash)

                        if translation and (not current_hu or force):
                            stats.records_checked += 1

                            if dry_run:
                                logger.debug(f"  [DRY RUN] Would update Neo4j translation: {code}")
                            else:
                                neo4j_session.run(
                                    "MATCH (n:DTCNode {code: $code}) SET n.description_hu = $hu",
                                    code=code,
                                    hu=translation
                                )
                            stats.records_updated += 1

            driver.close()

        except Exception as e:
            logger.warning(f"  Neo4j translation sync failed: {e}")

    except Exception as e:
        logger.error(f"Translation sync failed: {e}")
        stats.errors += 1

    stats.complete()
    return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Synchronize data between AutoCognitix databases"
    )

    # Sync options
    sync_group = parser.add_argument_group("Sync Operations")
    sync_group.add_argument(
        "--postgres-neo4j",
        action="store_true",
        help="Sync PostgreSQL to Neo4j",
    )
    sync_group.add_argument(
        "--postgres-qdrant",
        action="store_true",
        help="Sync PostgreSQL to Qdrant",
    )
    sync_group.add_argument(
        "--neo4j-postgres",
        action="store_true",
        help="Sync Neo4j to PostgreSQL",
    )
    sync_group.add_argument(
        "--translations",
        action="store_true",
        help="Sync translations across all databases",
    )
    sync_group.add_argument(
        "--all",
        action="store_true",
        help="Run all sync operations",
    )

    # Verification
    verify_group = parser.add_argument_group("Verification")
    verify_group.add_argument(
        "--verify",
        action="store_true",
        help="Verify data consistency (no changes made)",
    )
    verify_group.add_argument(
        "--report",
        type=str,
        metavar="FILE",
        help="Save verification report to file",
    )

    # Options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without applying",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force sync even for existing records",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for large syncs (default: 100)",
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

    if args.dry_run:
        logger.info("DRY RUN MODE - No changes will be made")

    # Handle verification
    if args.verify:
        report_path = Path(args.report) if args.report else None
        report = verify_consistency(report_path)
        sys.exit(0 if report.is_consistent else 1)

    # Check if any sync operation is specified
    if not any([args.postgres_neo4j, args.postgres_qdrant, args.neo4j_postgres, args.translations, args.all]):
        parser.error("At least one sync operation must be specified (or use --all)")

    results: List[SyncStats] = []

    try:
        if args.postgres_neo4j or args.all:
            stats = sync_postgres_to_neo4j(args.dry_run, args.force, args.batch_size)
            results.append(stats)

        if args.postgres_qdrant or args.all:
            stats = sync_postgres_to_qdrant(args.dry_run, args.force, args.batch_size)
            results.append(stats)

        if args.neo4j_postgres or args.all:
            stats = sync_neo4j_to_postgres(args.dry_run, args.force, args.batch_size)
            results.append(stats)

        if args.translations or args.all:
            stats = sync_translations(args.dry_run, args.force)
            results.append(stats)

        # Summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("SYNC SUMMARY")
        logger.info("=" * 60)
        for stats in results:
            logger.info(f"  {stats.source} -> {stats.target}:")
            logger.info(f"    Checked: {stats.records_checked}")
            logger.info(f"    Added:   {stats.records_added}")
            logger.info(f"    Updated: {stats.records_updated}")
            logger.info(f"    Skipped: {stats.records_skipped}")
            logger.info(f"    Errors:  {stats.errors}")
            logger.info(f"    Time:    {stats.duration_seconds:.2f}s")
            logger.info("")
        logger.info("=" * 60)

        # Exit with error if any sync had errors
        if any(s.errors > 0 for s in results):
            sys.exit(1)

    except Exception as e:
        logger.error(f"Sync failed: {e}")
        raise


if __name__ == "__main__":
    main()
