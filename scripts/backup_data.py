#!/usr/bin/env python3
"""
Comprehensive Backup Script for AutoCognitix

Creates backups of all data sources with support for:
- Full backups (complete database dump)
- Incremental backups (changes since last backup)
- Restore capability (from any backup)
- Backup verification (integrity checks)

Backup targets:
- PostgreSQL dump (pg_dump or Python fallback)
- Neo4j export (Cypher queries)
- Qdrant snapshot
- JSON data files compression

Usage:
    # Full backup
    python scripts/backup_data.py --all                    # Backup everything
    python scripts/backup_data.py --postgres               # PostgreSQL only
    python scripts/backup_data.py --neo4j                  # Neo4j only
    python scripts/backup_data.py --qdrant                 # Qdrant only
    python scripts/backup_data.py --json                   # JSON files only

    # Incremental backup
    python scripts/backup_data.py --incremental            # Backup changes since last backup
    python scripts/backup_data.py --incremental --since 2024-01-01  # Since specific date

    # Restore
    python scripts/backup_data.py --restore <backup_path>  # Restore from backup
    python scripts/backup_data.py --restore <backup_path> --target postgres  # Restore specific target

    # Verification
    python scripts/backup_data.py --verify <backup_path>   # Verify backup integrity

    # Management
    python scripts/backup_data.py --list                   # List all backups
    python scripts/backup_data.py --cleanup --keep 5       # Keep only last 5 backups
"""

import argparse
import gzip
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

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

# Backup directory
BACKUP_DIR = PROJECT_ROOT / "data" / "backups"
DATA_DIR = PROJECT_ROOT / "data"

# Backup metadata file
BACKUP_STATE_FILE = BACKUP_DIR / ".backup_state.json"


class BackupState:
    """Manages backup state for incremental backups."""

    def __init__(self):
        self.state_file = BACKUP_STATE_FILE
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        """Load backup state from file."""
        if self.state_file.exists():
            with open(self.state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            "last_full_backup": None,
            "last_incremental_backup": None,
            "backup_history": [],
        }

    def save(self) -> None:
        """Save backup state to file."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, indent=2, default=str)

    def record_backup(
        self,
        backup_path: Path,
        backup_type: str,
        targets: List[str],
        stats: Dict[str, Any],
    ) -> None:
        """Record a backup in the state."""
        backup_record = {
            "timestamp": datetime.now().isoformat(),
            "path": str(backup_path),
            "type": backup_type,
            "targets": targets,
            "stats": stats,
        }

        if backup_type == "full":
            self.state["last_full_backup"] = backup_record
        else:
            self.state["last_incremental_backup"] = backup_record

        self.state["backup_history"].append(backup_record)
        self.save()

    def get_last_backup_time(self) -> Optional[datetime]:
        """Get the timestamp of the last backup."""
        last_full = self.state.get("last_full_backup")
        last_incr = self.state.get("last_incremental_backup")

        times = []
        if last_full:
            times.append(datetime.fromisoformat(last_full["timestamp"]))
        if last_incr:
            times.append(datetime.fromisoformat(last_incr["timestamp"]))

        return max(times) if times else None

    def get_backup_history(self) -> List[Dict[str, Any]]:
        """Get backup history."""
        return self.state.get("backup_history", [])


def ensure_backup_dir(backup_type: str = "full") -> Path:
    """Create backup directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"{backup_type}_{timestamp}"
    backup_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Backup directory: {backup_path}")
    return backup_path


def parse_database_url(url: str) -> Dict[str, str]:
    """Parse database URL into components."""
    # Remove async prefix if present
    if url.startswith("postgresql+asyncpg://"):
        url = url.replace("postgresql+asyncpg://", "postgresql://")

    parsed = urlparse(url)
    return {
        "host": parsed.hostname or "localhost",
        "port": str(parsed.port or 5432),
        "user": parsed.username or "postgres",
        "password": parsed.password or "",
        "database": parsed.path.lstrip("/") or "autocognitix",
    }


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def backup_postgres(
    backup_path: Path,
    incremental: bool = False,
    since: Optional[datetime] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Create PostgreSQL backup using pg_dump.

    Returns tuple of (success, stats).
    """
    logger.info("Starting PostgreSQL backup...")
    stats: Dict[str, Any] = {"tables": 0, "rows": 0}

    try:
        db_config = parse_database_url(settings.DATABASE_URL)

        # Set environment for password
        env = os.environ.copy()
        if db_config["password"]:
            env["PGPASSWORD"] = db_config["password"]

        output_file = backup_path / "postgres_dump.sql"
        output_gz = backup_path / "postgres_dump.sql.gz"

        # Try pg_dump
        cmd = [
            "pg_dump",
            "-h", db_config["host"],
            "-p", db_config["port"],
            "-U", db_config["user"],
            "-d", db_config["database"],
            "-F", "p",  # Plain text format
            "--no-owner",
            "--no-acl",
            "-f", str(output_file),
        ]

        logger.info(f"Running: pg_dump to {output_file}")
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"pg_dump failed: {result.stderr}")
            # Try alternative: export via Python
            return backup_postgres_via_python(backup_path, incremental, since)

        # Compress the dump
        logger.info("Compressing PostgreSQL dump...")
        with open(output_file, 'rb') as f_in:
            with gzip.open(output_gz, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Remove uncompressed file
        output_file.unlink()

        # Compute hash for verification
        file_hash = compute_file_hash(output_gz)

        file_size = output_gz.stat().st_size / 1024
        stats["file_size_kb"] = round(file_size, 1)
        stats["hash"] = file_hash

        logger.info(f"PostgreSQL backup complete: {output_gz} ({file_size:.1f} KB)")
        return True, stats

    except FileNotFoundError:
        logger.warning("pg_dump not found, trying Python-based export...")
        return backup_postgres_via_python(backup_path, incremental, since)
    except Exception as e:
        logger.error(f"PostgreSQL backup failed: {e}")
        return False, stats


def backup_postgres_via_python(
    backup_path: Path,
    incremental: bool = False,
    since: Optional[datetime] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Backup PostgreSQL using Python SQLAlchemy (fallback).
    Exports data as JSON for portability.
    """
    logger.info("Using Python-based PostgreSQL export...")
    stats: Dict[str, Any] = {"tables": 0, "rows": 0}

    try:
        from sqlalchemy import create_engine, inspect, text
        from sqlalchemy.orm import Session

        # Convert async URL to sync
        url = settings.DATABASE_URL
        if url.startswith("postgresql+asyncpg://"):
            url = url.replace("postgresql+asyncpg://", "postgresql://")

        engine = create_engine(url)
        inspector = inspect(engine)

        backup_data: Dict[str, Any] = {
            "backup_time": datetime.now().isoformat(),
            "backup_type": "incremental" if incremental else "full",
            "since": since.isoformat() if since else None,
            "database": "autocognitix",
            "tables": {}
        }

        with Session(engine) as session:
            # Get all table names
            table_names = inspector.get_table_names()
            stats["tables"] = len(table_names)

            for table_name in table_names:
                # Security: Validate table name
                if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', table_name):
                    logger.warning(f"  Skipping invalid table name: {table_name}")
                    continue

                # Skip sensitive tables
                if table_name in ("alembic_version", "users"):
                    continue

                logger.info(f"  Exporting table: {table_name}")

                # Get columns
                columns = [col["name"] for col in inspector.get_columns(table_name)]

                # Build query with optional incremental filter
                query = f'SELECT * FROM "{table_name}"'
                if incremental and since and "updated_at" in columns:
                    query += f" WHERE updated_at >= '{since.isoformat()}'"
                elif incremental and since and "created_at" in columns:
                    query += f" WHERE created_at >= '{since.isoformat()}'"

                result = session.execute(text(query))
                rows = result.fetchall()

                # Convert to list of dicts
                table_data = []
                for row in rows:
                    row_dict = {}
                    for i, col in enumerate(columns):
                        value = row[i]
                        # Handle non-serializable types
                        if hasattr(value, 'isoformat'):
                            value = value.isoformat()
                        elif isinstance(value, bytes):
                            value = value.hex()
                        row_dict[col] = value
                    table_data.append(row_dict)

                backup_data["tables"][table_name] = {
                    "columns": columns,
                    "row_count": len(table_data),
                    "data": table_data,
                }
                stats["rows"] += len(table_data)

        # Save as compressed JSON
        output_file = backup_path / "postgres_backup.json.gz"
        with gzip.open(output_file, 'wt', encoding='utf-8') as f:
            json.dump(backup_data, f, indent=2, ensure_ascii=False, default=str)

        # Compute hash
        file_hash = compute_file_hash(output_file)

        file_size = output_file.stat().st_size / 1024
        stats["file_size_kb"] = round(file_size, 1)
        stats["hash"] = file_hash

        logger.info(f"PostgreSQL Python export complete: {output_file} ({file_size:.1f} KB)")
        return True, stats

    except Exception as e:
        logger.error(f"PostgreSQL Python export failed: {e}")
        return False, stats


def backup_neo4j(
    backup_path: Path,
    incremental: bool = False,
    since: Optional[datetime] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Create Neo4j backup using Cypher queries.

    Exports all nodes and relationships as JSON.
    """
    logger.info("Starting Neo4j backup...")
    stats: Dict[str, Any] = {"nodes": 0, "relationships": 0}

    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )

        backup_data: Dict[str, Any] = {
            "backup_time": datetime.now().isoformat(),
            "backup_type": "incremental" if incremental else "full",
            "since": since.isoformat() if since else None,
            "database": "neo4j",
            "nodes": {},
            "relationships": [],
        }

        with driver.session() as session:
            # Get all node labels
            labels_result = session.run("CALL db.labels()")
            labels = [record["label"] for record in labels_result]

            # Export nodes by label
            for label in labels:
                # Security: Validate label name
                if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', label):
                    logger.warning(f"  Skipping invalid label name: {label}")
                    continue

                logger.info(f"  Exporting nodes: {label}")
                result = session.run(f"MATCH (n:{label}) RETURN n, elementId(n) as id")
                nodes = []
                for record in result:
                    node = record["n"]
                    node_data = dict(node)
                    node_data["_element_id"] = record["id"]
                    nodes.append(node_data)
                backup_data["nodes"][label] = nodes
                stats["nodes"] += len(nodes)
                logger.info(f"    Found {len(nodes)} {label} nodes")

            # Export relationships
            logger.info("  Exporting relationships...")
            result = session.run("""
                MATCH (a)-[r]->(b)
                RETURN
                    labels(a)[0] as start_label,
                    elementId(a) as start_id,
                    type(r) as rel_type,
                    properties(r) as rel_props,
                    labels(b)[0] as end_label,
                    elementId(b) as end_id
            """)

            for record in result:
                backup_data["relationships"].append({
                    "start_label": record["start_label"],
                    "start_id": record["start_id"],
                    "type": record["rel_type"],
                    "properties": dict(record["rel_props"]) if record["rel_props"] else {},
                    "end_label": record["end_label"],
                    "end_id": record["end_id"],
                })

            stats["relationships"] = len(backup_data["relationships"])
            logger.info(f"    Found {stats['relationships']} relationships")

        driver.close()

        # Save as compressed JSON
        output_file = backup_path / "neo4j_backup.json.gz"
        with gzip.open(output_file, 'wt', encoding='utf-8') as f:
            json.dump(backup_data, f, indent=2, ensure_ascii=False, default=str)

        # Compute hash
        file_hash = compute_file_hash(output_file)

        file_size = output_file.stat().st_size / 1024
        stats["file_size_kb"] = round(file_size, 1)
        stats["hash"] = file_hash

        logger.info(f"Neo4j backup complete: {output_file} ({file_size:.1f} KB)")
        return True, stats

    except Exception as e:
        logger.error(f"Neo4j backup failed: {e}")
        return False, stats


def backup_qdrant(
    backup_path: Path,
    incremental: bool = False,
    since: Optional[datetime] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Create Qdrant snapshot/backup.

    Exports all collections and their vectors.
    """
    logger.info("Starting Qdrant backup...")
    stats: Dict[str, Any] = {"collections": 0, "points": 0}

    try:
        from qdrant_client import QdrantClient

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

        backup_data: Dict[str, Any] = {
            "backup_time": datetime.now().isoformat(),
            "backup_type": "incremental" if incremental else "full",
            "collections": {},
        }

        # Get all collections
        collections = client.get_collections().collections
        stats["collections"] = len(collections)

        for collection in collections:
            name = collection.name
            logger.info(f"  Exporting collection: {name}")

            # Get collection info
            info = client.get_collection(name)

            collection_data = {
                "vector_size": info.config.params.vectors.size if hasattr(info.config.params.vectors, 'size') else None,
                "distance": str(info.config.params.vectors.distance) if hasattr(info.config.params.vectors, 'distance') else None,
                "points_count": info.points_count,
                "points": [],
            }

            # Scroll through all points
            offset = None
            while True:
                result = client.scroll(
                    collection_name=name,
                    offset=offset,
                    limit=100,
                    with_payload=True,
                    with_vectors=True,
                )

                points, next_offset = result

                for point in points:
                    point_data = {
                        "id": point.id,
                        "payload": point.payload,
                        "vector": point.vector if isinstance(point.vector, list) else list(point.vector) if point.vector else None,
                    }
                    collection_data["points"].append(point_data)

                if next_offset is None:
                    break
                offset = next_offset

            backup_data["collections"][name] = collection_data
            stats["points"] += len(collection_data["points"])
            logger.info(f"    Exported {len(collection_data['points'])} points")

        # Save as compressed JSON
        output_file = backup_path / "qdrant_backup.json.gz"
        with gzip.open(output_file, 'wt', encoding='utf-8') as f:
            json.dump(backup_data, f, ensure_ascii=False, default=str)

        # Compute hash
        file_hash = compute_file_hash(output_file)

        file_size = output_file.stat().st_size / 1024
        stats["file_size_kb"] = round(file_size, 1)
        stats["hash"] = file_hash

        logger.info(f"Qdrant backup complete: {output_file} ({file_size:.1f} KB)")
        return True, stats

    except Exception as e:
        logger.error(f"Qdrant backup failed: {e}")
        return False, stats


def backup_json_files(
    backup_path: Path,
    incremental: bool = False,
    since: Optional[datetime] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Compress all JSON data files into a tarball.
    """
    logger.info("Starting JSON files backup...")
    stats: Dict[str, Any] = {"files": 0}

    try:
        output_file = backup_path / "json_data.tar.gz"

        # Find all JSON files in data directory
        json_files: List[Path] = []
        for pattern in ["*.json", "**/*.json"]:
            json_files.extend(DATA_DIR.glob(pattern))

        # Filter out files in backups directory
        json_files = [f for f in json_files if "backups" not in str(f)]

        # Filter by modification time for incremental backup
        if incremental and since:
            json_files = [
                f for f in json_files
                if datetime.fromtimestamp(f.stat().st_mtime) >= since
            ]

        if not json_files:
            logger.warning("No JSON files found to backup")
            return True, stats

        stats["files"] = len(json_files)
        logger.info(f"  Found {len(json_files)} JSON files")

        # Create tar.gz archive
        with tarfile.open(output_file, "w:gz") as tar:
            for json_file in json_files:
                # Use relative path from DATA_DIR
                arcname = json_file.relative_to(DATA_DIR)
                tar.add(json_file, arcname=arcname)
                logger.debug(f"    Added: {arcname}")

        # Compute hash
        file_hash = compute_file_hash(output_file)

        file_size = output_file.stat().st_size / 1024
        stats["file_size_kb"] = round(file_size, 1)
        stats["hash"] = file_hash

        logger.info(f"JSON backup complete: {output_file} ({file_size:.1f} KB)")
        return True, stats

    except Exception as e:
        logger.error(f"JSON backup failed: {e}")
        return False, stats


def create_backup_manifest(
    backup_path: Path,
    results: Dict[str, Tuple[bool, Dict[str, Any]]],
    backup_type: str = "full",
) -> None:
    """Create a manifest file with backup details."""
    manifest = {
        "backup_time": datetime.now().isoformat(),
        "backup_type": backup_type,
        "backup_path": str(backup_path),
        "results": {},
        "files": [],
        "verification": {},
    }

    # Add results with stats
    for target, (success, stats) in results.items():
        manifest["results"][target] = {
            "success": success,
            "stats": stats,
        }
        if "hash" in stats:
            manifest["verification"][target] = stats["hash"]

    # List all files
    for file in backup_path.iterdir():
        if file.is_file() and file.name != "manifest.json":
            manifest["files"].append({
                "name": file.name,
                "size_kb": round(file.stat().st_size / 1024, 1),
                "hash": compute_file_hash(file),
            })

    manifest_file = backup_path / "manifest.json"
    with open(manifest_file, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Manifest created: {manifest_file}")


def verify_backup(backup_path: Path) -> bool:
    """
    Verify backup integrity.

    Checks file hashes against manifest.
    """
    logger.info(f"Verifying backup: {backup_path}")

    manifest_file = backup_path / "manifest.json"
    if not manifest_file.exists():
        logger.error("Manifest file not found")
        return False

    with open(manifest_file, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    all_valid = True

    for file_info in manifest.get("files", []):
        file_path = backup_path / file_info["name"]

        if not file_path.exists():
            logger.error(f"  Missing file: {file_info['name']}")
            all_valid = False
            continue

        expected_hash = file_info.get("hash")
        if expected_hash:
            actual_hash = compute_file_hash(file_path)
            if actual_hash != expected_hash:
                logger.error(f"  Hash mismatch: {file_info['name']}")
                logger.error(f"    Expected: {expected_hash}")
                logger.error(f"    Actual:   {actual_hash}")
                all_valid = False
            else:
                logger.info(f"  Verified: {file_info['name']}")
        else:
            logger.warning(f"  No hash for: {file_info['name']}")

    if all_valid:
        logger.info("Backup verification PASSED")
    else:
        logger.error("Backup verification FAILED")

    return all_valid


def restore_backup(
    backup_path: Path,
    targets: Optional[List[str]] = None,
    dry_run: bool = False,
) -> bool:
    """
    Restore from a backup.

    Args:
        backup_path: Path to the backup directory
        targets: List of targets to restore (postgres, neo4j, qdrant, json)
        dry_run: If True, only show what would be restored
    """
    logger.info(f"Restoring from backup: {backup_path}")

    if dry_run:
        logger.info("DRY RUN - No changes will be made")

    manifest_file = backup_path / "manifest.json"
    if not manifest_file.exists():
        logger.error("Manifest file not found")
        return False

    with open(manifest_file, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    # Determine which targets to restore
    if targets is None:
        targets = list(manifest.get("results", {}).keys())

    logger.info(f"Targets to restore: {targets}")

    results = []

    for target in targets:
        target_result = manifest.get("results", {}).get(target)
        if not target_result or not target_result.get("success"):
            logger.warning(f"  Skipping {target}: backup was not successful")
            continue

        if target == "postgres":
            result = restore_postgres(backup_path, dry_run)
            results.append(("PostgreSQL", result))

        elif target == "neo4j":
            result = restore_neo4j(backup_path, dry_run)
            results.append(("Neo4j", result))

        elif target == "qdrant":
            result = restore_qdrant(backup_path, dry_run)
            results.append(("Qdrant", result))

        elif target == "json":
            result = restore_json_files(backup_path, dry_run)
            results.append(("JSON files", result))

    # Summary
    logger.info("=" * 50)
    logger.info("RESTORE SUMMARY")
    logger.info("=" * 50)
    for name, success in results:
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"  {name}: {status}")
    logger.info("=" * 50)

    return all(success for _, success in results)


def restore_postgres(backup_path: Path, dry_run: bool = False) -> bool:
    """Restore PostgreSQL from backup."""
    logger.info("  Restoring PostgreSQL...")

    # Check for SQL dump
    sql_file = backup_path / "postgres_dump.sql.gz"
    json_file = backup_path / "postgres_backup.json.gz"

    if sql_file.exists():
        if dry_run:
            logger.info("    Would restore from SQL dump")
            return True

        # Decompress and restore
        db_config = parse_database_url(settings.DATABASE_URL)
        env = os.environ.copy()
        if db_config["password"]:
            env["PGPASSWORD"] = db_config["password"]

        # Decompress
        temp_sql = backup_path / "postgres_dump.sql"
        with gzip.open(sql_file, 'rb') as f_in:
            with open(temp_sql, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Restore using psql
        cmd = [
            "psql",
            "-h", db_config["host"],
            "-p", db_config["port"],
            "-U", db_config["user"],
            "-d", db_config["database"],
            "-f", str(temp_sql),
        ]

        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        temp_sql.unlink()

        if result.returncode != 0:
            logger.error(f"    psql restore failed: {result.stderr}")
            return restore_postgres_from_json(json_file, dry_run)

        logger.info("    PostgreSQL restored from SQL dump")
        return True

    elif json_file.exists():
        return restore_postgres_from_json(json_file, dry_run)

    else:
        logger.error("    No PostgreSQL backup file found")
        return False


def restore_postgres_from_json(json_file: Path, dry_run: bool = False) -> bool:
    """Restore PostgreSQL from JSON backup."""
    if not json_file.exists():
        return False

    logger.info("    Restoring from JSON backup...")

    if dry_run:
        logger.info("    Would restore from JSON backup")
        return True

    try:
        from sqlalchemy import create_engine, text
        from sqlalchemy.orm import Session

        # Load backup data
        with gzip.open(json_file, 'rt', encoding='utf-8') as f:
            backup_data = json.load(f)

        # Convert async URL to sync
        url = settings.DATABASE_URL
        if url.startswith("postgresql+asyncpg://"):
            url = url.replace("postgresql+asyncpg://", "postgresql://")

        engine = create_engine(url)

        with Session(engine) as session:
            for table_name, table_info in backup_data.get("tables", {}).items():
                # Security: validate table name
                if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', table_name):
                    continue

                if table_name in ("users", "alembic_version"):
                    continue

                data_rows = table_info.get("data", [])
                if not data_rows:
                    continue

                logger.info(f"      Restoring table: {table_name} ({len(data_rows)} rows)")

                # Clear existing data
                session.execute(text(f'DELETE FROM "{table_name}"'))

                # Insert data
                columns = table_info.get("columns", list(data_rows[0].keys()))
                for row in data_rows:
                    cols = [f'"{c}"' for c in row.keys()]
                    placeholders = [f":{c.replace('.', '_')}" for c in row.keys()]
                    sql = f'INSERT INTO "{table_name}" ({", ".join(cols)}) VALUES ({", ".join(placeholders)})'

                    params = {}
                    for k, v in row.items():
                        param_name = k.replace('.', '_')
                        if isinstance(v, (list, dict)):
                            params[param_name] = json.dumps(v)
                        else:
                            params[param_name] = v

                    session.execute(text(sql), params)

                session.commit()

        logger.info("    PostgreSQL restored from JSON")
        return True

    except Exception as e:
        logger.error(f"    PostgreSQL JSON restore failed: {e}")
        return False


def restore_neo4j(backup_path: Path, dry_run: bool = False) -> bool:
    """Restore Neo4j from backup."""
    logger.info("  Restoring Neo4j...")

    json_file = backup_path / "neo4j_backup.json.gz"
    if not json_file.exists():
        logger.error("    No Neo4j backup file found")
        return False

    if dry_run:
        logger.info("    Would restore from JSON backup")
        return True

    try:
        from neo4j import GraphDatabase

        # Load backup data
        with gzip.open(json_file, 'rt', encoding='utf-8') as f:
            backup_data = json.load(f)

        driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )

        with driver.session() as session:
            # Restore nodes
            for label, nodes in backup_data.get("nodes", {}).items():
                # Security: validate label name
                if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', label):
                    continue

                logger.info(f"      Restoring {len(nodes)} {label} nodes...")
                for node in nodes:
                    # Remove internal fields
                    props = {k: v for k, v in node.items() if not k.startswith("_")}
                    props_clause = ", ".join([f"{k}: ${k}" for k in props.keys()])
                    session.run(
                        f"MERGE (n:{label} {{{props_clause}}})",
                        **props
                    )

            # Restore relationships
            relationships = backup_data.get("relationships", [])
            if relationships:
                logger.info(f"      Restoring {len(relationships)} relationships...")
                for rel in relationships:
                    # This is simplified - may need adjustment based on your schema
                    query = f"""
                        MATCH (a:{rel['start_label']})
                        MATCH (b:{rel['end_label']})
                        WHERE elementId(a) = $start_id AND elementId(b) = $end_id
                        MERGE (a)-[r:{rel['type']}]->(b)
                    """
                    try:
                        session.run(
                            query,
                            start_id=rel["start_id"],
                            end_id=rel["end_id"],
                        )
                    except Exception:
                        pass  # Relationship may already exist

        driver.close()
        logger.info("    Neo4j restored")
        return True

    except Exception as e:
        logger.error(f"    Neo4j restore failed: {e}")
        return False


def restore_qdrant(backup_path: Path, dry_run: bool = False) -> bool:
    """Restore Qdrant from backup."""
    logger.info("  Restoring Qdrant...")

    json_file = backup_path / "qdrant_backup.json.gz"
    if not json_file.exists():
        logger.error("    No Qdrant backup file found")
        return False

    if dry_run:
        logger.info("    Would restore from JSON backup")
        return True

    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import Distance, PointStruct, VectorParams

        # Load backup data
        with gzip.open(json_file, 'rt', encoding='utf-8') as f:
            backup_data = json.load(f)

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

        for name, collection_data in backup_data.get("collections", {}).items():
            logger.info(f"      Restoring collection: {name}")

            vector_size = collection_data.get("vector_size")
            distance = collection_data.get("distance", "Cosine")

            # Check if collection exists
            existing = [c.name for c in client.get_collections().collections]

            if name not in existing and vector_size:
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

            # Restore points in batches
            points = collection_data.get("points", [])
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                point_structs = []
                for point in batch:
                    if point.get("vector"):
                        point_structs.append(PointStruct(
                            id=point["id"],
                            vector=point["vector"],
                            payload=point.get("payload", {}),
                        ))
                if point_structs:
                    client.upsert(collection_name=name, points=point_structs)

            logger.info(f"        Restored {len(points)} points")

        logger.info("    Qdrant restored")
        return True

    except Exception as e:
        logger.error(f"    Qdrant restore failed: {e}")
        return False


def restore_json_files(backup_path: Path, dry_run: bool = False) -> bool:
    """Restore JSON files from backup."""
    logger.info("  Restoring JSON files...")

    tar_file = backup_path / "json_data.tar.gz"
    if not tar_file.exists():
        logger.error("    No JSON backup file found")
        return False

    if dry_run:
        logger.info("    Would restore from tar.gz archive")
        return True

    try:
        with tarfile.open(tar_file, "r:gz") as tar:
            # Extract to data directory
            tar.extractall(path=DATA_DIR)

        logger.info("    JSON files restored")
        return True

    except Exception as e:
        logger.error(f"    JSON files restore failed: {e}")
        return False


def list_backups() -> None:
    """List all available backups."""
    logger.info("Available backups:")

    if not BACKUP_DIR.exists():
        logger.info("  No backups found")
        return

    backups = []
    for path in BACKUP_DIR.iterdir():
        if path.is_dir() and not path.name.startswith("."):
            manifest_file = path / "manifest.json"
            if manifest_file.exists():
                with open(manifest_file, 'r', encoding='utf-8') as f:
                    manifest = json.load(f)
                backups.append({
                    "path": path,
                    "timestamp": manifest.get("backup_time"),
                    "type": manifest.get("backup_type", "unknown"),
                    "targets": list(manifest.get("results", {}).keys()),
                })

    if not backups:
        logger.info("  No backups found")
        return

    # Sort by timestamp
    backups.sort(key=lambda x: x["timestamp"] or "", reverse=True)

    for backup in backups:
        targets_str = ", ".join(backup["targets"])
        logger.info(f"  {backup['path'].name}")
        logger.info(f"    Time: {backup['timestamp']}")
        logger.info(f"    Type: {backup['type']}")
        logger.info(f"    Targets: {targets_str}")


def cleanup_backups(keep: int = 5) -> None:
    """Remove old backups, keeping only the most recent ones."""
    logger.info(f"Cleaning up backups, keeping {keep} most recent...")

    if not BACKUP_DIR.exists():
        logger.info("  No backups to clean up")
        return

    backups = []
    for path in BACKUP_DIR.iterdir():
        if path.is_dir() and not path.name.startswith("."):
            manifest_file = path / "manifest.json"
            if manifest_file.exists():
                with open(manifest_file, 'r', encoding='utf-8') as f:
                    manifest = json.load(f)
                backups.append({
                    "path": path,
                    "timestamp": manifest.get("backup_time"),
                })

    # Sort by timestamp (newest first)
    backups.sort(key=lambda x: x["timestamp"] or "", reverse=True)

    # Remove old backups
    removed = 0
    for backup in backups[keep:]:
        logger.info(f"  Removing: {backup['path'].name}")
        shutil.rmtree(backup["path"])
        removed += 1

    logger.info(f"Removed {removed} old backups")


# =============================================================================
# Cloud Storage Upload Functions
# =============================================================================

def create_backup_archive(backup_path: Path) -> Path:
    """
    Create a single compressed archive from the backup directory.

    Returns:
        Path to the created archive file.
    """
    archive_name = f"{backup_path.name}.tar.gz"
    archive_path = backup_path.parent / archive_name

    logger.info(f"Creating backup archive: {archive_path}")

    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(backup_path, arcname=backup_path.name)

    file_size = archive_path.stat().st_size / (1024 * 1024)  # MB
    logger.info(f"Archive created: {archive_path} ({file_size:.2f} MB)")

    return archive_path


def upload_to_s3(
    backup_path: Path,
    bucket: str,
    prefix: str = "backups",
    region: Optional[str] = None,
    endpoint_url: Optional[str] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Upload backup to AWS S3 or S3-compatible storage (MinIO, DigitalOcean Spaces, etc.).

    Requires environment variables:
    - AWS_ACCESS_KEY_ID
    - AWS_SECRET_ACCESS_KEY
    - AWS_DEFAULT_REGION (optional, can be passed as argument)

    For S3-compatible services, use endpoint_url parameter.

    Args:
        backup_path: Path to the backup directory or archive
        bucket: S3 bucket name
        prefix: Key prefix for the backup (default: "backups")
        region: AWS region (optional)
        endpoint_url: Custom endpoint URL for S3-compatible services

    Returns:
        Tuple of (success, upload_info)
    """
    logger.info(f"Uploading backup to S3: s3://{bucket}/{prefix}/")

    try:
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError
    except ImportError:
        logger.error("boto3 not installed. Install with: pip install boto3")
        return False, {"error": "boto3 not installed"}

    try:
        # Create S3 client
        client_kwargs: Dict[str, Any] = {}
        if region:
            client_kwargs["region_name"] = region
        if endpoint_url:
            client_kwargs["endpoint_url"] = endpoint_url

        s3_client = boto3.client("s3", **client_kwargs)

        # Create archive if backup_path is a directory
        if backup_path.is_dir():
            archive_path = create_backup_archive(backup_path)
            upload_file = archive_path
            cleanup_archive = True
        else:
            upload_file = backup_path
            cleanup_archive = False

        # Generate S3 key
        s3_key = f"{prefix}/{upload_file.name}"

        # Upload with progress callback
        file_size = upload_file.stat().st_size

        def progress_callback(bytes_transferred: int) -> None:
            percent = (bytes_transferred / file_size) * 100
            logger.debug(f"  Upload progress: {percent:.1f}%")

        logger.info(f"  Uploading {upload_file.name} ({file_size / (1024*1024):.2f} MB)...")

        s3_client.upload_file(
            str(upload_file),
            bucket,
            s3_key,
            Callback=progress_callback,
            ExtraArgs={
                "Metadata": {
                    "backup-time": datetime.now().isoformat(),
                    "source": "autocognitix",
                },
            },
        )

        # Cleanup temporary archive
        if cleanup_archive and archive_path.exists():
            archive_path.unlink()

        # Get uploaded object info
        response = s3_client.head_object(Bucket=bucket, Key=s3_key)

        upload_info = {
            "bucket": bucket,
            "key": s3_key,
            "size_bytes": response.get("ContentLength", 0),
            "etag": response.get("ETag", "").strip('"'),
            "upload_time": datetime.now().isoformat(),
            "s3_uri": f"s3://{bucket}/{s3_key}",
        }

        logger.info(f"S3 upload complete: s3://{bucket}/{s3_key}")
        return True, upload_info

    except NoCredentialsError:
        logger.error("AWS credentials not found. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
        return False, {"error": "credentials_not_found"}
    except ClientError as e:
        logger.error(f"S3 upload failed: {e}")
        return False, {"error": str(e)}
    except Exception as e:
        logger.error(f"S3 upload failed: {e}")
        return False, {"error": str(e)}


def upload_to_gcs(
    backup_path: Path,
    bucket: str,
    prefix: str = "backups",
    project: Optional[str] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Upload backup to Google Cloud Storage.

    Requires:
    - GOOGLE_APPLICATION_CREDENTIALS environment variable or default credentials

    Args:
        backup_path: Path to the backup directory or archive
        bucket: GCS bucket name
        prefix: Blob prefix for the backup (default: "backups")
        project: GCP project ID (optional)

    Returns:
        Tuple of (success, upload_info)
    """
    logger.info(f"Uploading backup to GCS: gs://{bucket}/{prefix}/")

    try:
        from google.cloud import storage
        from google.cloud.exceptions import GoogleCloudError
    except ImportError:
        logger.error("google-cloud-storage not installed. Install with: pip install google-cloud-storage")
        return False, {"error": "google-cloud-storage not installed"}

    try:
        # Create GCS client
        client_kwargs: Dict[str, Any] = {}
        if project:
            client_kwargs["project"] = project

        storage_client = storage.Client(**client_kwargs)
        bucket_obj = storage_client.bucket(bucket)

        # Create archive if backup_path is a directory
        if backup_path.is_dir():
            archive_path = create_backup_archive(backup_path)
            upload_file = archive_path
            cleanup_archive = True
        else:
            upload_file = backup_path
            cleanup_archive = False

        # Generate GCS blob name
        blob_name = f"{prefix}/{upload_file.name}"
        blob = bucket_obj.blob(blob_name)

        file_size = upload_file.stat().st_size
        logger.info(f"  Uploading {upload_file.name} ({file_size / (1024*1024):.2f} MB)...")

        # Upload with metadata
        blob.metadata = {
            "backup-time": datetime.now().isoformat(),
            "source": "autocognitix",
        }
        blob.upload_from_filename(str(upload_file))

        # Cleanup temporary archive
        if cleanup_archive and archive_path.exists():
            archive_path.unlink()

        upload_info = {
            "bucket": bucket,
            "blob": blob_name,
            "size_bytes": blob.size,
            "md5_hash": blob.md5_hash,
            "upload_time": datetime.now().isoformat(),
            "gs_uri": f"gs://{bucket}/{blob_name}",
        }

        logger.info(f"GCS upload complete: gs://{bucket}/{blob_name}")
        return True, upload_info

    except GoogleCloudError as e:
        logger.error(f"GCS upload failed: {e}")
        return False, {"error": str(e)}
    except Exception as e:
        logger.error(f"GCS upload failed: {e}")
        return False, {"error": str(e)}


def upload_to_azure(
    backup_path: Path,
    container: str,
    connection_string: Optional[str] = None,
    prefix: str = "backups",
) -> Tuple[bool, Dict[str, Any]]:
    """
    Upload backup to Azure Blob Storage.

    Requires environment variable:
    - AZURE_STORAGE_CONNECTION_STRING or pass connection_string argument

    Args:
        backup_path: Path to the backup directory or archive
        container: Azure container name
        connection_string: Azure storage connection string (optional, uses env var if not provided)
        prefix: Blob prefix for the backup (default: "backups")

    Returns:
        Tuple of (success, upload_info)
    """
    logger.info(f"Uploading backup to Azure: {container}/{prefix}/")

    try:
        from azure.storage.blob import BlobServiceClient
        from azure.core.exceptions import AzureError
    except ImportError:
        logger.error("azure-storage-blob not installed. Install with: pip install azure-storage-blob")
        return False, {"error": "azure-storage-blob not installed"}

    try:
        # Get connection string
        conn_str = connection_string or os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        if not conn_str:
            logger.error("Azure connection string not found. Set AZURE_STORAGE_CONNECTION_STRING")
            return False, {"error": "connection_string_not_found"}

        # Create blob service client
        blob_service_client = BlobServiceClient.from_connection_string(conn_str)
        container_client = blob_service_client.get_container_client(container)

        # Create container if it doesn't exist
        try:
            container_client.create_container()
        except AzureError:
            pass  # Container already exists

        # Create archive if backup_path is a directory
        if backup_path.is_dir():
            archive_path = create_backup_archive(backup_path)
            upload_file = archive_path
            cleanup_archive = True
        else:
            upload_file = backup_path
            cleanup_archive = False

        # Generate blob name
        blob_name = f"{prefix}/{upload_file.name}"
        blob_client = container_client.get_blob_client(blob_name)

        file_size = upload_file.stat().st_size
        logger.info(f"  Uploading {upload_file.name} ({file_size / (1024*1024):.2f} MB)...")

        # Upload with metadata
        with open(upload_file, "rb") as data:
            blob_client.upload_blob(
                data,
                overwrite=True,
                metadata={
                    "backup_time": datetime.now().isoformat(),
                    "source": "autocognitix",
                },
            )

        # Cleanup temporary archive
        if cleanup_archive and archive_path.exists():
            archive_path.unlink()

        # Get blob properties
        properties = blob_client.get_blob_properties()

        upload_info = {
            "container": container,
            "blob": blob_name,
            "size_bytes": properties.size,
            "etag": properties.etag,
            "upload_time": datetime.now().isoformat(),
            "azure_uri": f"https://{blob_service_client.account_name}.blob.core.windows.net/{container}/{blob_name}",
        }

        logger.info(f"Azure upload complete: {container}/{blob_name}")
        return True, upload_info

    except AzureError as e:
        logger.error(f"Azure upload failed: {e}")
        return False, {"error": str(e)}
    except Exception as e:
        logger.error(f"Azure upload failed: {e}")
        return False, {"error": str(e)}


def download_from_s3(
    bucket: str,
    key: str,
    output_path: Path,
    region: Optional[str] = None,
    endpoint_url: Optional[str] = None,
) -> Tuple[bool, Path]:
    """
    Download backup from S3.

    Args:
        bucket: S3 bucket name
        key: S3 object key
        output_path: Local path to save the backup
        region: AWS region (optional)
        endpoint_url: Custom endpoint URL for S3-compatible services

    Returns:
        Tuple of (success, downloaded_path)
    """
    logger.info(f"Downloading from S3: s3://{bucket}/{key}")

    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        logger.error("boto3 not installed")
        return False, output_path

    try:
        client_kwargs: Dict[str, Any] = {}
        if region:
            client_kwargs["region_name"] = region
        if endpoint_url:
            client_kwargs["endpoint_url"] = endpoint_url

        s3_client = boto3.client("s3", **client_kwargs)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Download file
        s3_client.download_file(bucket, key, str(output_path))

        logger.info(f"Downloaded to: {output_path}")
        return True, output_path

    except ClientError as e:
        logger.error(f"S3 download failed: {e}")
        return False, output_path


def list_cloud_backups(
    provider: str,
    bucket: str,
    prefix: str = "backups",
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """
    List backups in cloud storage.

    Args:
        provider: Cloud provider ("s3", "gcs", "azure")
        bucket: Bucket/container name
        prefix: Prefix to filter backups
        **kwargs: Provider-specific arguments

    Returns:
        List of backup info dictionaries
    """
    backups: List[Dict[str, Any]] = []

    if provider == "s3":
        try:
            import boto3

            client_kwargs: Dict[str, Any] = {}
            if kwargs.get("region"):
                client_kwargs["region_name"] = kwargs["region"]
            if kwargs.get("endpoint_url"):
                client_kwargs["endpoint_url"] = kwargs["endpoint_url"]

            s3_client = boto3.client("s3", **client_kwargs)

            response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)

            for obj in response.get("Contents", []):
                backups.append({
                    "provider": "s3",
                    "key": obj["Key"],
                    "size_bytes": obj["Size"],
                    "last_modified": obj["LastModified"].isoformat(),
                    "uri": f"s3://{bucket}/{obj['Key']}",
                })

        except Exception as e:
            logger.error(f"Failed to list S3 backups: {e}")

    elif provider == "gcs":
        try:
            from google.cloud import storage

            client = storage.Client()
            bucket_obj = client.bucket(bucket)

            for blob in bucket_obj.list_blobs(prefix=prefix):
                backups.append({
                    "provider": "gcs",
                    "key": blob.name,
                    "size_bytes": blob.size,
                    "last_modified": blob.updated.isoformat() if blob.updated else None,
                    "uri": f"gs://{bucket}/{blob.name}",
                })

        except Exception as e:
            logger.error(f"Failed to list GCS backups: {e}")

    elif provider == "azure":
        try:
            from azure.storage.blob import BlobServiceClient

            conn_str = kwargs.get("connection_string") or os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
            if conn_str:
                blob_service_client = BlobServiceClient.from_connection_string(conn_str)
                container_client = blob_service_client.get_container_client(bucket)

                for blob in container_client.list_blobs(name_starts_with=prefix):
                    backups.append({
                        "provider": "azure",
                        "key": blob.name,
                        "size_bytes": blob.size,
                        "last_modified": blob.last_modified.isoformat() if blob.last_modified else None,
                        "uri": f"azure://{bucket}/{blob.name}",
                    })

        except Exception as e:
            logger.error(f"Failed to list Azure backups: {e}")

    return backups


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Backup AutoCognitix data with full and incremental support"
    )

    # Backup targets
    target_group = parser.add_argument_group("Backup Targets")
    target_group.add_argument("--postgres", action="store_true", help="Backup PostgreSQL")
    target_group.add_argument("--neo4j", action="store_true", help="Backup Neo4j")
    target_group.add_argument("--qdrant", action="store_true", help="Backup Qdrant")
    target_group.add_argument("--json", action="store_true", help="Backup JSON files")
    target_group.add_argument("--all", action="store_true", help="Backup everything")

    # Backup modes
    mode_group = parser.add_argument_group("Backup Modes")
    mode_group.add_argument("--incremental", action="store_true", help="Incremental backup (changes since last backup)")
    mode_group.add_argument("--since", type=str, help="Incremental backup since date (YYYY-MM-DD)")

    # Restore
    restore_group = parser.add_argument_group("Restore Options")
    restore_group.add_argument("--restore", type=str, metavar="PATH", help="Restore from backup")
    restore_group.add_argument("--target", type=str, action="append", help="Target(s) to restore")
    restore_group.add_argument("--dry-run", action="store_true", help="Preview restore without changes")

    # Verification
    parser.add_argument("--verify", type=str, metavar="PATH", help="Verify backup integrity")

    # Management
    parser.add_argument("--list", action="store_true", help="List all backups")
    parser.add_argument("--cleanup", action="store_true", help="Remove old backups")
    parser.add_argument("--keep", type=int, default=5, help="Number of backups to keep (default: 5)")

    # Cloud Storage Upload
    cloud_group = parser.add_argument_group("Cloud Storage Upload")
    cloud_group.add_argument(
        "--upload-s3",
        type=str,
        metavar="BUCKET",
        help="Upload backup to AWS S3 bucket"
    )
    cloud_group.add_argument(
        "--upload-gcs",
        type=str,
        metavar="BUCKET",
        help="Upload backup to Google Cloud Storage bucket"
    )
    cloud_group.add_argument(
        "--upload-azure",
        type=str,
        metavar="CONTAINER",
        help="Upload backup to Azure Blob Storage container"
    )
    cloud_group.add_argument(
        "--s3-prefix",
        type=str,
        default="backups",
        help="S3/GCS/Azure prefix for backup (default: backups)"
    )
    cloud_group.add_argument(
        "--s3-region",
        type=str,
        help="AWS region for S3 upload"
    )
    cloud_group.add_argument(
        "--s3-endpoint",
        type=str,
        help="Custom S3 endpoint URL (for MinIO, DigitalOcean Spaces, etc.)"
    )
    cloud_group.add_argument(
        "--list-cloud",
        type=str,
        choices=["s3", "gcs", "azure"],
        help="List backups in cloud storage"
    )
    cloud_group.add_argument(
        "--cloud-bucket",
        type=str,
        help="Bucket/container name for --list-cloud"
    )

    # Other
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Handle list cloud backups
    if args.list_cloud:
        if not args.cloud_bucket:
            parser.error("--cloud-bucket is required with --list-cloud")

        backups = list_cloud_backups(
            provider=args.list_cloud,
            bucket=args.cloud_bucket,
            prefix=args.s3_prefix,
            region=args.s3_region,
            endpoint_url=args.s3_endpoint,
        )

        if backups:
            logger.info(f"Cloud backups in {args.list_cloud}://{args.cloud_bucket}/{args.s3_prefix}/")
            for backup in backups:
                size_mb = backup.get("size_bytes", 0) / (1024 * 1024)
                logger.info(f"  {backup['key']} ({size_mb:.2f} MB) - {backup.get('last_modified', 'unknown')}")
        else:
            logger.info("No cloud backups found")
        sys.exit(0)

    # Handle restore
    if args.restore:
        backup_path = Path(args.restore)
        if not backup_path.is_absolute():
            backup_path = BACKUP_DIR / args.restore
        success = restore_backup(backup_path, args.target, args.dry_run)
        sys.exit(0 if success else 1)

    # Handle verification
    if args.verify:
        backup_path = Path(args.verify)
        if not backup_path.is_absolute():
            backup_path = BACKUP_DIR / args.verify
        success = verify_backup(backup_path)
        sys.exit(0 if success else 1)

    # Handle list
    if args.list:
        list_backups()
        sys.exit(0)

    # Handle cleanup
    if args.cleanup:
        cleanup_backups(args.keep)
        sys.exit(0)

    # Default to --all if no specific backup is selected
    if not any([args.postgres, args.neo4j, args.qdrant, args.json, args.all]):
        args.all = True

    # Determine backup type
    backup_type = "incremental" if args.incremental else "full"

    # Parse since date
    since = None
    if args.since:
        since = datetime.fromisoformat(args.since)
    elif args.incremental:
        state = BackupState()
        since = state.get_last_backup_time()
        if since:
            logger.info(f"Incremental backup since: {since}")
        else:
            logger.info("No previous backup found, performing full backup")
            backup_type = "full"

    # Create backup directory
    backup_path = ensure_backup_dir(backup_type)

    results: Dict[str, Tuple[bool, Dict[str, Any]]] = {}
    targets_backed_up = []

    try:
        if args.postgres or args.all:
            success, stats = backup_postgres(backup_path, args.incremental, since)
            results["postgres"] = (success, stats)
            if success:
                targets_backed_up.append("postgres")

        if args.neo4j or args.all:
            success, stats = backup_neo4j(backup_path, args.incremental, since)
            results["neo4j"] = (success, stats)
            if success:
                targets_backed_up.append("neo4j")

        if args.qdrant or args.all:
            success, stats = backup_qdrant(backup_path, args.incremental, since)
            results["qdrant"] = (success, stats)
            if success:
                targets_backed_up.append("qdrant")

        if args.json or args.all:
            success, stats = backup_json_files(backup_path, args.incremental, since)
            results["json"] = (success, stats)
            if success:
                targets_backed_up.append("json")

        # Create manifest
        create_backup_manifest(backup_path, results, backup_type)

        # Record backup in state
        state = BackupState()
        all_stats = {k: v[1] for k, v in results.items()}
        state.record_backup(backup_path, backup_type, targets_backed_up, all_stats)

        # Summary
        logger.info("=" * 50)
        logger.info("BACKUP SUMMARY")
        logger.info("=" * 50)
        for name, (success, stats) in results.items():
            status = "SUCCESS" if success else "FAILED"
            size = stats.get("file_size_kb", "N/A")
            logger.info(f"  {name}: {status} ({size} KB)")
        logger.info(f"Backup location: {backup_path}")
        logger.info("=" * 50)

        # Cloud storage upload (if requested)
        cloud_upload_success = True

        if args.upload_s3:
            logger.info("")
            logger.info("Uploading to AWS S3...")
            success, upload_info = upload_to_s3(
                backup_path=backup_path,
                bucket=args.upload_s3,
                prefix=args.s3_prefix,
                region=args.s3_region,
                endpoint_url=args.s3_endpoint,
            )
            if success:
                logger.info(f"  S3 URI: {upload_info.get('s3_uri')}")
                # Update manifest with cloud info
                manifest_file = backup_path / "manifest.json"
                if manifest_file.exists():
                    with open(manifest_file, 'r', encoding='utf-8') as f:
                        manifest = json.load(f)
                    manifest["cloud_upload"] = {"s3": upload_info}
                    with open(manifest_file, 'w', encoding='utf-8') as f:
                        json.dump(manifest, f, indent=2)
            else:
                cloud_upload_success = False

        if args.upload_gcs:
            logger.info("")
            logger.info("Uploading to Google Cloud Storage...")
            success, upload_info = upload_to_gcs(
                backup_path=backup_path,
                bucket=args.upload_gcs,
                prefix=args.s3_prefix,
            )
            if success:
                logger.info(f"  GCS URI: {upload_info.get('gs_uri')}")
                # Update manifest with cloud info
                manifest_file = backup_path / "manifest.json"
                if manifest_file.exists():
                    with open(manifest_file, 'r', encoding='utf-8') as f:
                        manifest = json.load(f)
                    if "cloud_upload" not in manifest:
                        manifest["cloud_upload"] = {}
                    manifest["cloud_upload"]["gcs"] = upload_info
                    with open(manifest_file, 'w', encoding='utf-8') as f:
                        json.dump(manifest, f, indent=2)
            else:
                cloud_upload_success = False

        if args.upload_azure:
            logger.info("")
            logger.info("Uploading to Azure Blob Storage...")
            success, upload_info = upload_to_azure(
                backup_path=backup_path,
                container=args.upload_azure,
                prefix=args.s3_prefix,
            )
            if success:
                logger.info(f"  Azure URI: {upload_info.get('azure_uri')}")
                # Update manifest with cloud info
                manifest_file = backup_path / "manifest.json"
                if manifest_file.exists():
                    with open(manifest_file, 'r', encoding='utf-8') as f:
                        manifest = json.load(f)
                    if "cloud_upload" not in manifest:
                        manifest["cloud_upload"] = {}
                    manifest["cloud_upload"]["azure"] = upload_info
                    with open(manifest_file, 'w', encoding='utf-8') as f:
                        json.dump(manifest, f, indent=2)
            else:
                cloud_upload_success = False

        # Exit with error if any backup or cloud upload failed
        if not all(success for success, _ in results.values()) or not cloud_upload_success:
            sys.exit(1)

    except Exception as e:
        logger.error(f"Backup failed: {e}")
        raise


if __name__ == "__main__":
    main()
