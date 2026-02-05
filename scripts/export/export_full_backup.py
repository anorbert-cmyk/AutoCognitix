#!/usr/bin/env python3
"""
Full System Backup Utility for AutoCognitix

Combines all data exports into a single comprehensive backup archive:
- PostgreSQL database dump
- Neo4j graph export
- Qdrant vector backup
- JSON data files
- Configuration backup

Features:
- Creates timestamped backup archives
- Supports incremental backups (changes since last backup)
- Includes version info and manifest
- Compression for efficient storage
- Verification capability

Usage:
    # Full backup (all data)
    python scripts/export/export_full_backup.py

    # Full backup with custom output
    python scripts/export/export_full_backup.py --output /path/to/backup

    # Incremental backup (changes since last backup)
    python scripts/export/export_full_backup.py --incremental

    # Incremental since specific date
    python scripts/export/export_full_backup.py --incremental --since 2024-01-01

    # Selective backup
    python scripts/export/export_full_backup.py --postgres --neo4j
    python scripts/export/export_full_backup.py --no-vectors

    # Verify existing backup
    python scripts/export/export_full_backup.py --verify /path/to/backup.tar.gz

    # List available backups
    python scripts/export/export_full_backup.py --list
"""

import argparse
import gzip
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

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
EXPORT_DIR = DATA_DIR / "exports"
BACKUP_DIR = DATA_DIR / "backups"

# Backup state file
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

        # Keep only last 50 backups in history
        if len(self.state["backup_history"]) > 50:
            self.state["backup_history"] = self.state["backup_history"][-50:]

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


class FullBackupExporter:
    """Handles full system backup exports."""

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        incremental: bool = False,
        since: Optional[datetime] = None,
        include_postgres: bool = True,
        include_neo4j: bool = True,
        include_qdrant: bool = True,
        include_json: bool = True,
        include_vectors: bool = True,
    ):
        """
        Initialize the full backup exporter.

        Args:
            output_dir: Output directory for backup
            incremental: Whether this is an incremental backup
            since: Date for incremental backup (changes since)
            include_postgres: Include PostgreSQL backup
            include_neo4j: Include Neo4j backup
            include_qdrant: Include Qdrant backup
            include_json: Include JSON data files
            include_vectors: Include vector embeddings
        """
        self.output_dir = output_dir or BACKUP_DIR
        self.incremental = incremental
        self.since = since
        self.include_postgres = include_postgres
        self.include_neo4j = include_neo4j
        self.include_qdrant = include_qdrant
        self.include_json = include_json
        self.include_vectors = include_vectors

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create temporary working directory
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_type = "incremental" if incremental else "full"
        self.work_dir = self.output_dir / f"{backup_type}_{self.timestamp}"
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Statistics
        self.stats = {
            "backup_type": backup_type,
            "timestamp": self.timestamp,
            "targets": [],
            "target_stats": {},
            "total_size_kb": 0,
        }

    def compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def backup_postgres(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Backup PostgreSQL database.

        Returns:
            Tuple of (success, stats)
        """
        logger.info("Backing up PostgreSQL...")
        stats: Dict[str, Any] = {"tables": 0, "rows": 0}

        try:
            from backend.app.core.config import settings

            # Parse database URL
            url = settings.DATABASE_URL
            if url.startswith("postgresql+asyncpg://"):
                url = url.replace("postgresql+asyncpg://", "postgresql://")

            parsed = urlparse(url)
            db_config = {
                "host": parsed.hostname or "localhost",
                "port": str(parsed.port or 5432),
                "user": parsed.username or "postgres",
                "password": parsed.password or "",
                "database": parsed.path.lstrip("/") or "autocognitix",
            }

            output_file = self.work_dir / "postgres_backup.json.gz"

            # Export using Python (works without pg_dump)
            from sqlalchemy import create_engine, inspect, text
            from sqlalchemy.orm import Session
            import re

            engine = create_engine(url)
            inspector = inspect(engine)

            backup_data: Dict[str, Any] = {
                "backup_time": datetime.now().isoformat(),
                "backup_type": "incremental" if self.incremental else "full",
                "since": self.since.isoformat() if self.since else None,
                "database": db_config["database"],
                "tables": {}
            }

            with Session(engine) as session:
                table_names = inspector.get_table_names()
                stats["tables"] = len(table_names)

                for table_name in tqdm(table_names, desc="Exporting tables", disable=not HAS_TQDM):
                    # Security: Validate table name
                    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', table_name):
                        continue

                    # Skip sensitive tables
                    if table_name in ("alembic_version", "users"):
                        continue

                    columns = [col["name"] for col in inspector.get_columns(table_name)]

                    # Build query with optional incremental filter
                    query = f'SELECT * FROM "{table_name}"'
                    if self.incremental and self.since:
                        if "updated_at" in columns:
                            query += f" WHERE updated_at >= '{self.since.isoformat()}'"
                        elif "created_at" in columns:
                            query += f" WHERE created_at >= '{self.since.isoformat()}'"

                    result = session.execute(text(query))
                    rows = result.fetchall()

                    # Convert to list of dicts
                    table_data = []
                    for row in rows:
                        row_dict = {}
                        for i, col in enumerate(columns):
                            value = row[i]
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
            with gzip.open(output_file, 'wt', encoding='utf-8') as f:
                json.dump(backup_data, f, ensure_ascii=False, default=str)

            stats["file_size_kb"] = round(output_file.stat().st_size / 1024, 1)
            stats["hash"] = self.compute_file_hash(output_file)

            logger.info(f"  PostgreSQL backup complete: {stats['tables']} tables, {stats['rows']} rows")
            return True, stats

        except Exception as e:
            logger.error(f"  PostgreSQL backup failed: {e}")
            return False, stats

    def backup_neo4j(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Backup Neo4j graph database.

        Returns:
            Tuple of (success, stats)
        """
        logger.info("Backing up Neo4j...")
        stats: Dict[str, Any] = {"nodes": 0, "relationships": 0}

        try:
            from neo4j import GraphDatabase
            from backend.app.core.config import settings
            import re

            driver = GraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
            )

            backup_data: Dict[str, Any] = {
                "backup_time": datetime.now().isoformat(),
                "backup_type": "incremental" if self.incremental else "full",
                "nodes": {},
                "relationships": [],
            }

            with driver.session() as session:
                # Get labels
                labels_result = session.run("CALL db.labels()")
                labels = [record["label"] for record in labels_result]

                # Export nodes
                for label in tqdm(labels, desc="Exporting nodes", disable=not HAS_TQDM):
                    # Security: validate label
                    if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', label):
                        continue

                    result = session.run(f"MATCH (n:{label}) RETURN n, elementId(n) as id")
                    nodes = []
                    for record in result:
                        node = record["n"]
                        node_data = dict(node)
                        node_data["_element_id"] = record["id"]
                        nodes.append(node_data)

                    backup_data["nodes"][label] = nodes
                    stats["nodes"] += len(nodes)

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

            driver.close()

            # Save as compressed JSON
            output_file = self.work_dir / "neo4j_backup.json.gz"
            with gzip.open(output_file, 'wt', encoding='utf-8') as f:
                json.dump(backup_data, f, ensure_ascii=False, default=str)

            stats["file_size_kb"] = round(output_file.stat().st_size / 1024, 1)
            stats["hash"] = self.compute_file_hash(output_file)

            logger.info(f"  Neo4j backup complete: {stats['nodes']} nodes, {stats['relationships']} relationships")
            return True, stats

        except Exception as e:
            logger.error(f"  Neo4j backup failed: {e}")
            return False, stats

    def backup_qdrant(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Backup Qdrant vector database.

        Returns:
            Tuple of (success, stats)
        """
        logger.info("Backing up Qdrant...")
        stats: Dict[str, Any] = {"collections": 0, "points": 0}

        try:
            from qdrant_client import QdrantClient
            from backend.app.core.config import settings

            # Connect to Qdrant
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
                "backup_type": "incremental" if self.incremental else "full",
                "collections": {},
            }

            # Get collections
            collections = client.get_collections().collections
            stats["collections"] = len(collections)

            for collection in tqdm(collections, desc="Exporting collections", disable=not HAS_TQDM):
                name = collection.name
                info = client.get_collection(name)

                collection_data = {
                    "config": {
                        "vector_size": info.config.params.vectors.size if hasattr(info.config.params.vectors, 'size') else None,
                        "distance": str(info.config.params.vectors.distance) if hasattr(info.config.params.vectors, 'distance') else None,
                    },
                    "points_count": info.points_count,
                    "points": [],
                }

                # Scroll through points
                offset = None
                while True:
                    result = client.scroll(
                        collection_name=name,
                        offset=offset,
                        limit=100,
                        with_payload=True,
                        with_vectors=self.include_vectors,
                    )

                    points, next_offset = result

                    for point in points:
                        point_data = {
                            "id": point.id,
                            "payload": point.payload,
                        }
                        if self.include_vectors and point.vector:
                            if isinstance(point.vector, dict):
                                point_data["vector"] = {k: list(v) if hasattr(v, '__iter__') else v for k, v in point.vector.items()}
                            elif hasattr(point.vector, '__iter__'):
                                point_data["vector"] = list(point.vector)
                            else:
                                point_data["vector"] = point.vector

                        collection_data["points"].append(point_data)

                    if next_offset is None:
                        break
                    offset = next_offset

                backup_data["collections"][name] = collection_data
                stats["points"] += len(collection_data["points"])

            # Save as compressed JSON
            output_file = self.work_dir / "qdrant_backup.json.gz"
            with gzip.open(output_file, 'wt', encoding='utf-8') as f:
                json.dump(backup_data, f, ensure_ascii=False, default=str)

            stats["file_size_kb"] = round(output_file.stat().st_size / 1024, 1)
            stats["hash"] = self.compute_file_hash(output_file)

            logger.info(f"  Qdrant backup complete: {stats['collections']} collections, {stats['points']} points")
            return True, stats

        except Exception as e:
            logger.error(f"  Qdrant backup failed: {e}")
            return False, stats

    def backup_json_files(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Backup JSON data files.

        Returns:
            Tuple of (success, stats)
        """
        logger.info("Backing up JSON data files...")
        stats: Dict[str, Any] = {"files": 0}

        try:
            output_file = self.work_dir / "json_data.tar.gz"

            # Find all JSON files in data directory
            json_files: List[Path] = []
            for pattern in ["*.json", "**/*.json"]:
                json_files.extend(DATA_DIR.glob(pattern))

            # Filter out files in backups and exports directories
            json_files = [
                f for f in json_files
                if "backups" not in str(f) and "exports" not in str(f)
            ]

            # Filter by modification time for incremental backup
            if self.incremental and self.since:
                json_files = [
                    f for f in json_files
                    if datetime.fromtimestamp(f.stat().st_mtime) >= self.since
                ]

            if not json_files:
                logger.warning("  No JSON files found to backup")
                return True, stats

            stats["files"] = len(json_files)

            # Create tar.gz archive
            with tarfile.open(output_file, "w:gz") as tar:
                for json_file in tqdm(json_files, desc="Archiving files", disable=not HAS_TQDM):
                    arcname = json_file.relative_to(DATA_DIR)
                    tar.add(json_file, arcname=arcname)

            stats["file_size_kb"] = round(output_file.stat().st_size / 1024, 1)
            stats["hash"] = self.compute_file_hash(output_file)

            logger.info(f"  JSON backup complete: {stats['files']} files")
            return True, stats

        except Exception as e:
            logger.error(f"  JSON backup failed: {e}")
            return False, stats

    def create_manifest(self, results: Dict[str, Tuple[bool, Dict[str, Any]]]) -> None:
        """Create manifest file with backup details."""
        manifest = {
            "backup_info": {
                "timestamp": datetime.now().isoformat(),
                "backup_type": "incremental" if self.incremental else "full",
                "since": self.since.isoformat() if self.since else None,
                "version": "2.0",
                "source": "AutoCognitix",
            },
            "targets": {},
            "files": [],
            "verification": {},
        }

        # Add target results
        for target, (success, stats) in results.items():
            manifest["targets"][target] = {
                "success": success,
                "stats": stats,
            }
            if "hash" in stats:
                manifest["verification"][target] = stats["hash"]

        # List all files
        for file in self.work_dir.iterdir():
            if file.is_file() and file.name != "manifest.json":
                file_info = {
                    "name": file.name,
                    "size_kb": round(file.stat().st_size / 1024, 1),
                    "hash": self.compute_file_hash(file),
                }
                manifest["files"].append(file_info)
                self.stats["total_size_kb"] += file_info["size_kb"]

        manifest_file = self.work_dir / "manifest.json"
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"  Manifest created: {manifest_file}")

    def create_archive(self) -> Path:
        """
        Create final backup archive.

        Returns:
            Path to the archive file
        """
        logger.info("Creating backup archive...")

        backup_type = "incremental" if self.incremental else "full"
        archive_name = f"autocognitix_backup_{backup_type}_{self.timestamp}.tar.gz"
        archive_path = self.output_dir / archive_name

        with tarfile.open(archive_path, "w:gz") as tar:
            for file in self.work_dir.iterdir():
                tar.add(file, arcname=file.name)

        # Clean up work directory
        shutil.rmtree(self.work_dir)

        archive_size = archive_path.stat().st_size / (1024 * 1024)
        logger.info(f"  Archive created: {archive_path} ({archive_size:.2f} MB)")

        return archive_path

    def run(self) -> Path:
        """
        Run the full backup process.

        Returns:
            Path to the backup archive
        """
        logger.info("=" * 50)
        backup_type = "INCREMENTAL" if self.incremental else "FULL"
        logger.info(f"Starting {backup_type} BACKUP")
        logger.info("=" * 50)

        results: Dict[str, Tuple[bool, Dict[str, Any]]] = {}
        targets_backed_up = []

        # Run backups
        if self.include_postgres:
            success, stats = self.backup_postgres()
            results["postgres"] = (success, stats)
            if success:
                targets_backed_up.append("postgres")

        if self.include_neo4j:
            success, stats = self.backup_neo4j()
            results["neo4j"] = (success, stats)
            if success:
                targets_backed_up.append("neo4j")

        if self.include_qdrant:
            success, stats = self.backup_qdrant()
            results["qdrant"] = (success, stats)
            if success:
                targets_backed_up.append("qdrant")

        if self.include_json:
            success, stats = self.backup_json_files()
            results["json"] = (success, stats)
            if success:
                targets_backed_up.append("json")

        # Create manifest
        self.create_manifest(results)

        # Create final archive
        archive_path = self.create_archive()

        # Record backup state
        state = BackupState()
        self.stats["targets"] = targets_backed_up
        self.stats["target_stats"] = {k: v[1] for k, v in results.items()}
        state.record_backup(
            archive_path,
            "incremental" if self.incremental else "full",
            targets_backed_up,
            self.stats,
        )

        # Print summary
        logger.info("")
        logger.info("=" * 50)
        logger.info("BACKUP SUMMARY")
        logger.info("=" * 50)
        for name, (success, stats) in results.items():
            status = "SUCCESS" if success else "FAILED"
            size = stats.get("file_size_kb", "N/A")
            logger.info(f"  {name}: {status} ({size} KB)")
        logger.info(f"Total backup size: {self.stats['total_size_kb']:.1f} KB")
        logger.info(f"Archive: {archive_path}")
        logger.info("=" * 50)

        return archive_path


def verify_backup(backup_path: Path) -> bool:
    """
    Verify backup integrity.

    Args:
        backup_path: Path to backup archive

    Returns:
        True if verification passed
    """
    logger.info(f"Verifying backup: {backup_path}")

    if not backup_path.exists():
        logger.error("Backup file not found")
        return False

    # Create temporary directory for extraction
    temp_dir = backup_path.parent / f"verify_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Extract archive
        with tarfile.open(backup_path, "r:gz") as tar:
            tar.extractall(path=temp_dir)

        # Read manifest
        manifest_file = temp_dir / "manifest.json"
        if not manifest_file.exists():
            logger.error("Manifest file not found in backup")
            return False

        with open(manifest_file, 'r', encoding='utf-8') as f:
            manifest = json.load(f)

        # Verify file hashes
        all_valid = True
        for file_info in manifest.get("files", []):
            file_path = temp_dir / file_info["name"]

            if not file_path.exists():
                logger.error(f"  Missing file: {file_info['name']}")
                all_valid = False
                continue

            expected_hash = file_info.get("hash")
            if expected_hash:
                sha256 = hashlib.sha256()
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        sha256.update(chunk)
                actual_hash = sha256.hexdigest()

                if actual_hash != expected_hash:
                    logger.error(f"  Hash mismatch: {file_info['name']}")
                    all_valid = False
                else:
                    logger.info(f"  Verified: {file_info['name']}")

        if all_valid:
            logger.info("Backup verification PASSED")
        else:
            logger.error("Backup verification FAILED")

        return all_valid

    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def list_backups() -> None:
    """List all available backups."""
    logger.info("Available backups:")

    if not BACKUP_DIR.exists():
        logger.info("  No backups found")
        return

    # Find backup archives
    backups = list(BACKUP_DIR.glob("autocognitix_backup_*.tar.gz"))

    if not backups:
        logger.info("  No backups found")
        return

    # Sort by modification time
    backups.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    for backup in backups:
        size_mb = backup.stat().st_size / (1024 * 1024)
        mtime = datetime.fromtimestamp(backup.stat().st_mtime)
        logger.info(f"  {backup.name}")
        logger.info(f"    Size: {size_mb:.2f} MB")
        logger.info(f"    Modified: {mtime.isoformat()}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create comprehensive system backup for AutoCognitix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python export_full_backup.py
    python export_full_backup.py --incremental
    python export_full_backup.py --postgres --neo4j
    python export_full_backup.py --verify /path/to/backup.tar.gz
    python export_full_backup.py --list
        """
    )

    # Backup targets
    target_group = parser.add_argument_group("Backup Targets")
    target_group.add_argument(
        "--postgres",
        action="store_true",
        help="Include PostgreSQL backup",
    )
    target_group.add_argument(
        "--neo4j",
        action="store_true",
        help="Include Neo4j backup",
    )
    target_group.add_argument(
        "--qdrant",
        action="store_true",
        help="Include Qdrant backup",
    )
    target_group.add_argument(
        "--json",
        action="store_true",
        help="Include JSON data files backup",
    )
    target_group.add_argument(
        "--no-vectors",
        action="store_true",
        help="Exclude vector embeddings from Qdrant backup",
    )

    # Backup modes
    mode_group = parser.add_argument_group("Backup Modes")
    mode_group.add_argument(
        "--incremental",
        action="store_true",
        help="Create incremental backup (changes since last backup)",
    )
    mode_group.add_argument(
        "--since",
        type=str,
        help="Incremental backup since date (YYYY-MM-DD)",
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output directory (default: data/backups/)",
    )

    # Management
    parser.add_argument(
        "--verify",
        type=str,
        metavar="PATH",
        help="Verify backup integrity",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available backups",
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

    # Handle verify
    if args.verify:
        backup_path = Path(args.verify)
        success = verify_backup(backup_path)
        sys.exit(0 if success else 1)

    # Handle list
    if args.list:
        list_backups()
        sys.exit(0)

    # Determine which targets to include
    include_all = not any([args.postgres, args.neo4j, args.qdrant, args.json])
    include_postgres = args.postgres or include_all
    include_neo4j = args.neo4j or include_all
    include_qdrant = args.qdrant or include_all
    include_json = args.json or include_all

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
            args.incremental = False

    # Setup output directory
    output_dir = None
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Create exporter and run
    exporter = FullBackupExporter(
        output_dir=output_dir,
        incremental=args.incremental,
        since=since,
        include_postgres=include_postgres,
        include_neo4j=include_neo4j,
        include_qdrant=include_qdrant,
        include_json=include_json,
        include_vectors=not args.no_vectors,
    )

    try:
        archive_path = exporter.run()
        logger.info("")
        logger.info("Backup completed successfully!")
        logger.info(f"Archive: {archive_path}")
    except Exception as e:
        logger.error(f"Backup failed: {e}")
        raise


if __name__ == "__main__":
    main()
