#!/usr/bin/env python3
"""
Qdrant Vector Export Utility for AutoCognitix

Exports Qdrant vector embeddings and metadata to multiple formats:
- NumPy format (.npy) - Vector arrays for machine learning
- JSON format - Metadata and payloads
- Full backup format - Vectors with metadata for restore

Supports exporting from collections:
- dtc_embeddings_hu - DTC code embeddings (Hungarian huBERT, 768-dim)
- symptom_embeddings_hu - Symptom text embeddings
- component_embeddings_hu - Vehicle component embeddings
- repair_embeddings_hu - Repair procedure embeddings
- known_issue_embeddings_hu - Known issue embeddings

Usage:
    # Export all collections
    python scripts/export/export_qdrant_vectors.py --all

    # Export specific format
    python scripts/export/export_qdrant_vectors.py --numpy
    python scripts/export/export_qdrant_vectors.py --json
    python scripts/export/export_qdrant_vectors.py --full

    # Export specific collections
    python scripts/export/export_qdrant_vectors.py --all --collections dtc_embeddings_hu

    # Export only metadata (no vectors)
    python scripts/export/export_qdrant_vectors.py --metadata-only
"""

import argparse
import gzip
import json
import logging
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

# Try to import numpy
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    logger.warning("numpy not installed. NumPy export will be disabled.")

# Directories
DATA_DIR = PROJECT_ROOT / "data"
EXPORT_DIR = DATA_DIR / "exports"

# Qdrant collection names used in AutoCognitix
COLLECTION_NAMES = [
    "dtc_embeddings_hu",
    "symptom_embeddings_hu",
    "component_embeddings_hu",
    "repair_embeddings_hu",
    "known_issue_embeddings_hu",
    # Legacy collections
    "dtc_embeddings",
    "symptom_embeddings",
    "known_issue_embeddings",
]


class QdrantExporter:
    """Handles exporting Qdrant vectors and metadata."""

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        collection_filter: Optional[List[str]] = None,
        include_vectors: bool = True,
        batch_size: int = 100,
    ):
        """
        Initialize the Qdrant exporter.

        Args:
            output_dir: Output directory for exports
            collection_filter: Filter by collection names
            include_vectors: Whether to include vector data
            batch_size: Batch size for scrolling through points
        """
        self.output_dir = output_dir or EXPORT_DIR
        self.collection_filter = collection_filter
        self.include_vectors = include_vectors
        self.batch_size = batch_size

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data containers
        self.collections: Dict[str, Dict[str, Any]] = {}

        # Statistics
        self.stats = {
            "total_collections": 0,
            "total_points": 0,
            "total_vectors": 0,
            "collections_info": {},
        }

    def connect(self):
        """
        Connect to Qdrant.

        Returns:
            Qdrant client instance
        """
        try:
            from qdrant_client import QdrantClient
            from backend.app.core.config import settings

            # Support both local Qdrant and Qdrant Cloud
            if settings.QDRANT_URL:
                client = QdrantClient(
                    url=settings.QDRANT_URL,
                    api_key=settings.QDRANT_API_KEY,
                )
                logger.info(f"Connected to Qdrant Cloud: {settings.QDRANT_URL}")
            else:
                client = QdrantClient(
                    host=settings.QDRANT_HOST,
                    port=settings.QDRANT_PORT,
                )
                logger.info(f"Connected to local Qdrant: {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")

            return client

        except ImportError:
            logger.error("qdrant-client package not installed. Run: pip install qdrant-client")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

    def load_data(self) -> None:
        """Load all collection data from Qdrant."""
        logger.info("Loading data from Qdrant...")

        client = self.connect()

        # Get available collections
        available_collections = [c.name for c in client.get_collections().collections]
        logger.info(f"Available collections: {available_collections}")

        # Determine which collections to export
        collections_to_export = self.collection_filter or available_collections

        for collection_name in collections_to_export:
            if collection_name not in available_collections:
                logger.warning(f"Collection '{collection_name}' not found, skipping")
                continue

            logger.info(f"  Loading collection: {collection_name}")

            try:
                # Get collection info
                info = client.get_collection(collection_name)

                collection_data = {
                    "name": collection_name,
                    "points_count": info.points_count,
                    "vectors_count": info.vectors_count,
                    "vector_size": None,
                    "distance": None,
                    "points": [],
                }

                # Extract vector config
                if hasattr(info.config.params, 'vectors'):
                    vectors_config = info.config.params.vectors
                    if hasattr(vectors_config, 'size'):
                        collection_data["vector_size"] = vectors_config.size
                    if hasattr(vectors_config, 'distance'):
                        collection_data["distance"] = str(vectors_config.distance)

                # Scroll through all points
                offset = None
                points_loaded = 0

                while True:
                    result = client.scroll(
                        collection_name=collection_name,
                        offset=offset,
                        limit=self.batch_size,
                        with_payload=True,
                        with_vectors=self.include_vectors,
                    )

                    points, next_offset = result

                    for point in points:
                        point_data = {
                            "id": point.id,
                            "payload": self._sanitize_payload(point.payload),
                        }

                        if self.include_vectors and point.vector is not None:
                            # Handle different vector formats
                            if isinstance(point.vector, dict):
                                # Named vectors
                                point_data["vector"] = {
                                    k: list(v) if hasattr(v, '__iter__') else v
                                    for k, v in point.vector.items()
                                }
                            elif hasattr(point.vector, '__iter__'):
                                point_data["vector"] = list(point.vector)
                            else:
                                point_data["vector"] = point.vector

                        collection_data["points"].append(point_data)
                        points_loaded += 1

                    if next_offset is None:
                        break
                    offset = next_offset

                    # Progress update
                    if points_loaded % 1000 == 0:
                        logger.info(f"    Loaded {points_loaded} points...")

                self.collections[collection_name] = collection_data

                # Update stats
                self.stats["collections_info"][collection_name] = {
                    "points_count": len(collection_data["points"]),
                    "vector_size": collection_data["vector_size"],
                    "distance": collection_data["distance"],
                }
                self.stats["total_points"] += len(collection_data["points"])
                if self.include_vectors:
                    self.stats["total_vectors"] += len(collection_data["points"])

                logger.info(f"    Loaded {len(collection_data['points'])} points")

            except Exception as e:
                logger.error(f"    Error loading collection {collection_name}: {e}")
                continue

        self.stats["total_collections"] = len(self.collections)
        logger.info(f"Total loaded: {self.stats['total_collections']} collections, {self.stats['total_points']} points")

    def _sanitize_payload(self, payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Sanitize payload data for export.

        Removes sensitive information and handles special types.
        """
        if not payload:
            return {}

        sanitized = {}
        sensitive_keys = {
            "password", "secret", "token", "api_key", "credentials",
        }

        for key, value in payload.items():
            # Skip sensitive fields
            if any(s in key.lower() for s in sensitive_keys):
                continue

            # Handle special types
            if hasattr(value, 'isoformat'):
                value = value.isoformat()
            elif isinstance(value, bytes):
                value = value.hex()

            sanitized[key] = value

        return sanitized

    def export_to_numpy(self) -> List[Path]:
        """
        Export vectors to NumPy format (.npy).

        Creates separate .npy files for each collection's vectors.

        Returns:
            List of paths to exported files
        """
        if not HAS_NUMPY:
            logger.error("NumPy is not installed. Cannot export to NumPy format.")
            return []

        if not self.include_vectors:
            logger.warning("Vectors were not loaded. Cannot export to NumPy format.")
            return []

        logger.info("Exporting to NumPy format...")

        exported_files = []

        # Create numpy subdirectory
        numpy_dir = self.output_dir / "numpy"
        numpy_dir.mkdir(exist_ok=True)

        for collection_name, collection_data in self.collections.items():
            points = collection_data["points"]
            if not points:
                continue

            # Extract vectors
            vectors = []
            point_ids = []

            for point in tqdm(points, desc=f"Processing {collection_name}", disable=not HAS_TQDM):
                if "vector" in point and point["vector"] is not None:
                    vector = point["vector"]
                    if isinstance(vector, dict):
                        # Named vectors - use first vector
                        vector = list(vector.values())[0] if vector else None
                    if vector is not None:
                        vectors.append(vector)
                        point_ids.append(point["id"])

            if not vectors:
                logger.warning(f"  No vectors found in collection {collection_name}")
                continue

            # Convert to numpy array
            vectors_array = np.array(vectors, dtype=np.float32)

            # Save vectors
            vectors_file = numpy_dir / f"{collection_name}_vectors.npy"
            np.save(vectors_file, vectors_array)

            # Save point IDs
            ids_file = numpy_dir / f"{collection_name}_ids.npy"
            np.save(ids_file, np.array(point_ids))

            logger.info(f"  {collection_name}: {vectors_array.shape[0]} vectors x {vectors_array.shape[1]} dimensions")
            exported_files.extend([vectors_file, ids_file])

        logger.info(f"NumPy export complete: {len(exported_files)} files in {numpy_dir}")

        return exported_files

    def export_to_json(self, include_vectors: bool = False) -> Path:
        """
        Export metadata (and optionally vectors) to JSON format.

        Args:
            include_vectors: Whether to include vectors in JSON

        Returns:
            Path to exported file
        """
        logger.info("Exporting to JSON format...")

        export_data = {
            "export_info": {
                "export_time": datetime.now().isoformat(),
                "source": "AutoCognitix",
                "version": "2.0",
                "include_vectors": include_vectors,
                "statistics": self.stats,
            },
            "collections": {},
        }

        for collection_name, collection_data in self.collections.items():
            logger.info(f"  Processing collection: {collection_name}")

            collection_export = {
                "name": collection_name,
                "vector_size": collection_data["vector_size"],
                "distance": collection_data["distance"],
                "points_count": len(collection_data["points"]),
                "points": [],
            }

            for point in tqdm(collection_data["points"], desc=f"Processing {collection_name}", disable=not HAS_TQDM):
                point_export = {
                    "id": point["id"],
                    "payload": point["payload"],
                }

                if include_vectors and "vector" in point:
                    point_export["vector"] = point["vector"]

                collection_export["points"].append(point_export)

            export_data["collections"][collection_name] = collection_export

        # Choose filename based on content
        if include_vectors:
            output_file = self.output_dir / "qdrant_full_export.json"
        else:
            output_file = self.output_dir / "qdrant_metadata.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)

        file_size = output_file.stat().st_size / 1024
        logger.info(f"JSON export complete: {output_file} ({file_size:.1f} KB)")

        return output_file

    def export_full_backup(self) -> Path:
        """
        Export full backup (vectors + metadata) in compressed format.

        This format is suitable for complete backup and restore.

        Returns:
            Path to exported file
        """
        if not self.include_vectors:
            logger.warning("Vectors were not loaded. Full backup will only contain metadata.")

        logger.info("Exporting full backup...")

        export_data = {
            "export_info": {
                "export_time": datetime.now().isoformat(),
                "source": "AutoCognitix",
                "version": "2.0",
                "type": "full_backup",
                "statistics": self.stats,
            },
            "collections": {},
        }

        for collection_name, collection_data in self.collections.items():
            logger.info(f"  Processing collection: {collection_name}")

            collection_export = {
                "config": {
                    "vector_size": collection_data["vector_size"],
                    "distance": collection_data["distance"],
                },
                "points": [],
            }

            for point in tqdm(collection_data["points"], desc=f"Processing {collection_name}", disable=not HAS_TQDM):
                point_export = {
                    "id": point["id"],
                    "payload": point["payload"],
                }

                if "vector" in point:
                    point_export["vector"] = point["vector"]

                collection_export["points"].append(point_export)

            export_data["collections"][collection_name] = collection_export

        # Save as compressed JSON
        output_file = self.output_dir / "qdrant_backup.json.gz"
        with gzip.open(output_file, 'wt', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, default=str)

        file_size = output_file.stat().st_size / 1024
        logger.info(f"Full backup export complete: {output_file} ({file_size:.1f} KB)")

        return output_file

    def export_collection_stats(self) -> Path:
        """
        Export collection statistics and analysis.

        Returns:
            Path to exported file
        """
        logger.info("Exporting collection statistics...")

        stats_data = {
            "export_time": datetime.now().isoformat(),
            "summary": self.stats,
            "collections": {},
        }

        for collection_name, collection_data in self.collections.items():
            collection_stats = {
                "name": collection_name,
                "vector_size": collection_data["vector_size"],
                "distance": collection_data["distance"],
                "points_count": len(collection_data["points"]),
                "payload_fields": {},
                "sample_payloads": [],
            }

            # Analyze payload fields
            payload_fields: Dict[str, int] = {}
            for point in collection_data["points"]:
                for key in point.get("payload", {}).keys():
                    payload_fields[key] = payload_fields.get(key, 0) + 1

            collection_stats["payload_fields"] = payload_fields

            # Get sample payloads (first 5)
            for point in collection_data["points"][:5]:
                collection_stats["sample_payloads"].append(point.get("payload", {}))

            # Vector statistics (if available)
            if self.include_vectors and HAS_NUMPY:
                vectors = []
                for point in collection_data["points"]:
                    if "vector" in point and point["vector"] is not None:
                        vector = point["vector"]
                        if isinstance(vector, dict):
                            vector = list(vector.values())[0] if vector else None
                        if vector is not None:
                            vectors.append(vector)

                if vectors:
                    vectors_array = np.array(vectors, dtype=np.float32)
                    collection_stats["vector_stats"] = {
                        "count": len(vectors),
                        "dimensions": vectors_array.shape[1] if len(vectors_array.shape) > 1 else 0,
                        "mean_norm": float(np.mean(np.linalg.norm(vectors_array, axis=1))),
                        "min_norm": float(np.min(np.linalg.norm(vectors_array, axis=1))),
                        "max_norm": float(np.max(np.linalg.norm(vectors_array, axis=1))),
                    }

            stats_data["collections"][collection_name] = collection_stats

        output_file = self.output_dir / "qdrant_stats.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, indent=2, ensure_ascii=False)

        file_size = output_file.stat().st_size / 1024
        logger.info(f"Statistics export complete: {output_file} ({file_size:.1f} KB)")

        return output_file

    def print_statistics(self) -> None:
        """Print export statistics."""
        logger.info("=" * 50)
        logger.info("EXPORT STATISTICS")
        logger.info("=" * 50)
        logger.info(f"Total collections: {self.stats['total_collections']}")
        logger.info(f"Total points: {self.stats['total_points']}")
        if self.include_vectors:
            logger.info(f"Total vectors: {self.stats['total_vectors']}")
        logger.info("")
        logger.info("Collections:")
        for name, info in self.stats["collections_info"].items():
            logger.info(f"  {name}:")
            logger.info(f"    Points: {info['points_count']}")
            if info['vector_size']:
                logger.info(f"    Vector size: {info['vector_size']}")
            if info['distance']:
                logger.info(f"    Distance: {info['distance']}")
        logger.info("=" * 50)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Export Qdrant vectors and metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python export_qdrant_vectors.py --all
    python export_qdrant_vectors.py --numpy --json
    python export_qdrant_vectors.py --all --collections dtc_embeddings_hu
    python export_qdrant_vectors.py --metadata-only
        """
    )

    # Format options
    format_group = parser.add_argument_group("Export Formats")
    format_group.add_argument(
        "--numpy",
        action="store_true",
        help="Export vectors to NumPy format (.npy files)",
    )
    format_group.add_argument(
        "--json",
        action="store_true",
        help="Export metadata to JSON format",
    )
    format_group.add_argument(
        "--full",
        action="store_true",
        help="Export full backup (vectors + metadata, compressed)",
    )
    format_group.add_argument(
        "--stats",
        action="store_true",
        help="Export collection statistics",
    )
    format_group.add_argument(
        "--all",
        action="store_true",
        help="Export to all formats",
    )

    # Filter options
    filter_group = parser.add_argument_group("Filtering Options")
    filter_group.add_argument(
        "--collections",
        type=str,
        help=f"Filter by collection names (comma-separated). Available: {', '.join(COLLECTION_NAMES)}",
    )
    filter_group.add_argument(
        "--metadata-only",
        action="store_true",
        help="Export only metadata (no vectors)",
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
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for loading points (default: 100)",
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

    # Parse collection filter
    collection_filter = None
    if args.collections:
        collection_filter = [c.strip() for c in args.collections.split(",")]

    # Setup output directory
    output_dir = None
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Determine if vectors should be loaded
    include_vectors = not args.metadata_only

    # Create exporter
    exporter = QdrantExporter(
        output_dir=output_dir,
        collection_filter=collection_filter,
        include_vectors=include_vectors,
        batch_size=args.batch_size,
    )

    # Load data
    try:
        exporter.load_data()
    except Exception as e:
        logger.error(f"Failed to load data from Qdrant: {e}")
        sys.exit(1)

    if exporter.stats["total_collections"] == 0:
        logger.warning("No collections found in Qdrant!")
        sys.exit(1)

    # Default to --all if no specific format is selected
    if not any([args.numpy, args.json, args.full, args.stats, args.all]):
        args.all = True

    # Export to requested formats
    exported_files = []

    try:
        if args.numpy or args.all:
            if include_vectors and HAS_NUMPY:
                exported_files.extend(exporter.export_to_numpy())
            elif not HAS_NUMPY:
                logger.warning("Skipping NumPy export - numpy not installed")
            else:
                logger.warning("Skipping NumPy export - vectors not loaded")

        if args.json or args.all:
            exported_files.append(exporter.export_to_json(include_vectors=False))

        if args.full or args.all:
            exported_files.append(exporter.export_full_backup())

        if args.stats or args.all:
            exported_files.append(exporter.export_collection_stats())

        # Print statistics
        exporter.print_statistics()

        # Summary
        logger.info("")
        logger.info("Exported files:")
        for f in exported_files:
            if f:
                logger.info(f"  {f}")

        logger.info("")
        logger.info("Export completed successfully!")

    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise


if __name__ == "__main__":
    main()
