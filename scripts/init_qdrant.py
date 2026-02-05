#!/usr/bin/env python3
"""
Qdrant Vector Database Initialization Script for AutoCognitix.

This script initializes all required Qdrant collections with proper
configuration for the AutoCognitix diagnostic system.

Usage:
    python scripts/init_qdrant.py                    # Initialize collections
    python scripts/init_qdrant.py --verify           # Verify collections exist
    python scripts/init_qdrant.py --recreate         # Drop and recreate all
    python scripts/init_qdrant.py --info             # Show collection info

Collections:
    - dtc_embeddings_hu: Hungarian DTC code descriptions (768-dim huBERT)
    - symptom_embeddings_hu: Hungarian symptom descriptions
    - known_issue_embeddings_hu: Known issue descriptions

Requirements:
    - Qdrant running locally or Qdrant Cloud configured
    - Environment variables: QDRANT_HOST, QDRANT_PORT, QDRANT_URL, QDRANT_API_KEY
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qdrant_models
    from qdrant_client.http.exceptions import UnexpectedResponse
except ImportError:
    print("Error: qdrant-client not installed. Run: pip install qdrant-client")
    sys.exit(1)

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

# Connection settings (from environment or defaults)
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_URL = os.getenv("QDRANT_URL")  # For Qdrant Cloud
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # For Qdrant Cloud

# Embedding dimensions
HUBERT_DIMENSION = 768  # Hungarian huBERT (SZTAKI-HLT/hubert-base-cc)
MINILM_DIMENSION = 384  # Sentence-transformers all-MiniLM-L6-v2 (legacy)

# =============================================================================
# Collection Definitions
# =============================================================================

COLLECTIONS: List[Dict[str, Any]] = [
    # Primary Hungarian collections (768-dim huBERT)
    {
        "name": "dtc_embeddings_hu",
        "description": "Hungarian DTC code descriptions with huBERT embeddings",
        "vector_size": HUBERT_DIMENSION,
        "distance": qdrant_models.Distance.COSINE,
        "payload_schema": {
            "code": {"type": "keyword", "description": "DTC code (e.g., P0300)"},
            "description_hu": {"type": "text", "description": "Hungarian description"},
            "description_en": {"type": "text", "description": "English description"},
            "category": {"type": "keyword", "description": "DTC category"},
            "severity": {"type": "keyword", "description": "Severity level"},
            "system": {"type": "keyword", "description": "Vehicle system"},
            "symptoms": {"type": "keyword[]", "description": "Associated symptoms"},
            "possible_causes": {"type": "keyword[]", "description": "Possible causes"},
            "diagnostic_steps": {"type": "keyword[]", "description": "Diagnostic steps"},
            "related_codes": {"type": "keyword[]", "description": "Related DTC codes"},
            "is_generic": {"type": "bool", "description": "Generic vs manufacturer-specific"},
        },
        "indexes": [
            {"field": "code", "type": "keyword"},
            {"field": "category", "type": "keyword"},
            {"field": "severity", "type": "keyword"},
            {"field": "system", "type": "keyword"},
            {"field": "is_generic", "type": "bool"},
        ],
    },
    {
        "name": "symptom_embeddings_hu",
        "description": "Hungarian symptom descriptions with huBERT embeddings",
        "vector_size": HUBERT_DIMENSION,
        "distance": qdrant_models.Distance.COSINE,
        "payload_schema": {
            "symptom_text": {"type": "text", "description": "Symptom description"},
            "related_dtc_codes": {"type": "keyword[]", "description": "Related DTC codes"},
            "related_codes_count": {"type": "integer", "description": "Number of related codes"},
            "vehicle_make": {"type": "keyword", "description": "Vehicle make (optional)"},
            "vehicle_model": {"type": "keyword", "description": "Vehicle model (optional)"},
        },
        "indexes": [
            {"field": "related_codes_count", "type": "integer"},
            {"field": "vehicle_make", "type": "keyword"},
        ],
    },
    {
        "name": "known_issue_embeddings_hu",
        "description": "Known issue descriptions with huBERT embeddings",
        "vector_size": HUBERT_DIMENSION,
        "distance": qdrant_models.Distance.COSINE,
        "payload_schema": {
            "title": {"type": "text", "description": "Issue title"},
            "description": {"type": "text", "description": "Issue description"},
            "symptoms": {"type": "keyword[]", "description": "Associated symptoms"},
            "related_dtc_codes": {"type": "keyword[]", "description": "Related DTC codes"},
            "applicable_makes": {"type": "keyword[]", "description": "Applicable vehicle makes"},
            "applicable_models": {"type": "keyword[]", "description": "Applicable vehicle models"},
            "year_start": {"type": "integer", "description": "Start year"},
            "year_end": {"type": "integer", "description": "End year"},
            "confidence": {"type": "float", "description": "Confidence score"},
            "source_type": {"type": "keyword", "description": "Data source type"},
        },
        "indexes": [
            {"field": "confidence", "type": "float"},
            {"field": "source_type", "type": "keyword"},
            {"field": "year_start", "type": "integer"},
            {"field": "year_end", "type": "integer"},
        ],
    },
    # Legacy English collections (384-dim, for backwards compatibility)
    {
        "name": "dtc_embeddings",
        "description": "Legacy English DTC embeddings (all-MiniLM-L6-v2)",
        "vector_size": MINILM_DIMENSION,
        "distance": qdrant_models.Distance.COSINE,
        "payload_schema": {},
        "indexes": [],
        "legacy": True,
    },
    {
        "name": "symptom_embeddings",
        "description": "Legacy English symptom embeddings (all-MiniLM-L6-v2)",
        "vector_size": MINILM_DIMENSION,
        "distance": qdrant_models.Distance.COSINE,
        "payload_schema": {},
        "indexes": [],
        "legacy": True,
    },
]


# =============================================================================
# Qdrant Client
# =============================================================================

def get_qdrant_client() -> QdrantClient:
    """Create Qdrant client based on configuration."""
    if QDRANT_URL:
        # Qdrant Cloud
        logger.info(f"Connecting to Qdrant Cloud: {QDRANT_URL}")
        return QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )
    else:
        # Local Qdrant
        logger.info(f"Connecting to local Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")
        return QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
        )


# =============================================================================
# Collection Management
# =============================================================================

def collection_exists(client: QdrantClient, name: str) -> bool:
    """Check if a collection exists."""
    try:
        collections = client.get_collections().collections
        return any(c.name == name for c in collections)
    except Exception as e:
        logger.error(f"Error checking collection existence: {e}")
        return False


def create_collection(
    client: QdrantClient,
    config: Dict[str, Any],
    recreate: bool = False,
) -> bool:
    """
    Create a collection with the specified configuration.

    Args:
        client: Qdrant client instance.
        config: Collection configuration dictionary.
        recreate: If True, delete existing collection first.

    Returns:
        True if collection was created/exists, False on error.
    """
    name = config["name"]
    vector_size = config["vector_size"]
    distance = config["distance"]

    try:
        exists = collection_exists(client, name)

        if exists:
            if recreate:
                logger.info(f"Deleting existing collection: {name}")
                client.delete_collection(collection_name=name)
            else:
                logger.info(f"Collection already exists: {name}")
                return True

        # Create the collection
        logger.info(f"Creating collection: {name} (vectors: {vector_size}-dim, distance: {distance})")

        client.create_collection(
            collection_name=name,
            vectors_config=qdrant_models.VectorParams(
                size=vector_size,
                distance=distance,
            ),
            # Optimizers config for better performance
            optimizers_config=qdrant_models.OptimizersConfigDiff(
                indexing_threshold=10000,  # Start indexing after 10k points
            ),
            # HNSW config for vector search
            hnsw_config=qdrant_models.HnswConfigDiff(
                m=16,  # Number of edges per node in graph
                ef_construct=100,  # Number of neighbors for construction
            ),
        )

        # Create payload indexes
        indexes = config.get("indexes", [])
        for index_config in indexes:
            field = index_config["field"]
            field_type = index_config["type"]

            schema_type = _get_schema_type(field_type)
            if schema_type:
                try:
                    client.create_payload_index(
                        collection_name=name,
                        field_name=field,
                        field_schema=schema_type,
                    )
                    logger.info(f"  Created payload index: {field} ({field_type})")
                except Exception as e:
                    logger.warning(f"  Could not create index for {field}: {e}")

        logger.info(f"Successfully created collection: {name}")
        return True

    except Exception as e:
        logger.error(f"Error creating collection {name}: {e}")
        return False


def _get_schema_type(field_type: str) -> Optional[qdrant_models.PayloadSchemaType]:
    """Map field type string to Qdrant schema type."""
    type_mapping = {
        "keyword": qdrant_models.PayloadSchemaType.KEYWORD,
        "integer": qdrant_models.PayloadSchemaType.INTEGER,
        "float": qdrant_models.PayloadSchemaType.FLOAT,
        "bool": qdrant_models.PayloadSchemaType.BOOL,
        "geo": qdrant_models.PayloadSchemaType.GEO,
        "text": qdrant_models.PayloadSchemaType.TEXT,
    }
    return type_mapping.get(field_type)


def get_collection_info(client: QdrantClient, name: str) -> Dict[str, Any]:
    """Get detailed information about a collection."""
    try:
        info = client.get_collection(collection_name=name)
        return {
            "name": name,
            "status": str(info.status),
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "config": {
                "vector_size": info.config.params.vectors.size if hasattr(info.config.params.vectors, 'size') else "multi",
                "distance": str(info.config.params.vectors.distance) if hasattr(info.config.params.vectors, 'distance') else "unknown",
            },
        }
    except Exception:
        return {"name": name, "status": "not_found"}


# =============================================================================
# CLI Commands
# =============================================================================

def cmd_init(recreate: bool = False, skip_legacy: bool = True) -> bool:
    """Initialize all collections."""
    client = get_qdrant_client()

    success_count = 0
    total_count = 0

    for config in COLLECTIONS:
        # Skip legacy collections by default
        if config.get("legacy") and skip_legacy:
            logger.info(f"Skipping legacy collection: {config['name']}")
            continue

        total_count += 1
        if create_collection(client, config, recreate=recreate):
            success_count += 1

    logger.info(f"\nInitialization complete: {success_count}/{total_count} collections")
    return success_count == total_count


def cmd_verify() -> bool:
    """Verify all collections exist and are healthy."""
    client = get_qdrant_client()

    print("\n" + "=" * 70)
    print("QDRANT COLLECTION VERIFICATION")
    print("=" * 70)

    all_ok = True

    for config in COLLECTIONS:
        name = config["name"]
        is_legacy = config.get("legacy", False)

        exists = collection_exists(client, name)
        status = "OK" if exists else "MISSING"
        legacy_tag = " [LEGACY]" if is_legacy else ""

        if exists:
            info = get_collection_info(client, name)
            print(f"\n[{status}] {name}{legacy_tag}")
            print(f"    Status: {info.get('status', 'unknown')}")
            print(f"    Points: {info.get('points_count', 0)}")
            print(f"    Vectors: {info.get('vectors_count', 0)}")
            print(f"    Indexed: {info.get('indexed_vectors_count', 0)}")
        else:
            if not is_legacy:
                all_ok = False
            print(f"\n[{status}] {name}{legacy_tag}")

    print("\n" + "=" * 70)

    if all_ok:
        print("All required collections are present and healthy.")
    else:
        print("Some collections are missing. Run: python scripts/init_qdrant.py")

    return all_ok


def cmd_info() -> None:
    """Show detailed information about all collections."""
    client = get_qdrant_client()

    print("\n" + "=" * 70)
    print("QDRANT COLLECTION INFORMATION")
    print("=" * 70)

    for config in COLLECTIONS:
        name = config["name"]
        description = config.get("description", "")
        is_legacy = config.get("legacy", False)

        print(f"\nCollection: {name}")
        print(f"Description: {description}")
        print(f"Legacy: {is_legacy}")
        print(f"Expected vector size: {config['vector_size']}")

        if collection_exists(client, name):
            info = get_collection_info(client, name)
            print(f"Status: {info.get('status', 'unknown')}")
            print(f"Points count: {info.get('points_count', 0)}")
            print(f"Vectors count: {info.get('vectors_count', 0)}")
            print(f"Indexed vectors: {info.get('indexed_vectors_count', 0)}")

            # Show payload indexes
            try:
                collection_info = client.get_collection(collection_name=name)
                if collection_info.payload_schema:
                    print("Payload indexes:")
                    for field, schema in collection_info.payload_schema.items():
                        print(f"  - {field}: {schema}")
            except Exception:
                pass
        else:
            print("Status: NOT FOUND")

    print("\n" + "=" * 70)


def cmd_drop(collection_name: Optional[str] = None) -> bool:
    """Drop one or all collections."""
    client = get_qdrant_client()

    if collection_name:
        # Drop specific collection
        if collection_exists(client, collection_name):
            client.delete_collection(collection_name=collection_name)
            logger.info(f"Dropped collection: {collection_name}")
            return True
        else:
            logger.warning(f"Collection not found: {collection_name}")
            return False
    else:
        # Drop all collections
        for config in COLLECTIONS:
            name = config["name"]
            if collection_exists(client, name):
                client.delete_collection(collection_name=name)
                logger.info(f"Dropped collection: {name}")

        logger.info("All collections dropped")
        return True


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Initialize Qdrant vector database for AutoCognitix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/init_qdrant.py              # Initialize all collections
    python scripts/init_qdrant.py --verify     # Verify collections exist
    python scripts/init_qdrant.py --recreate   # Drop and recreate all
    python scripts/init_qdrant.py --info       # Show collection details
    python scripts/init_qdrant.py --drop       # Drop all collections
        """,
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify collections exist, don't create",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate all collections",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show detailed collection information",
    )
    parser.add_argument(
        "--drop",
        action="store_true",
        help="Drop all collections (dangerous!)",
    )
    parser.add_argument(
        "--include-legacy",
        action="store_true",
        help="Include legacy collections in operations",
    )
    parser.add_argument(
        "--collection",
        type=str,
        help="Operate on specific collection only",
    )

    args = parser.parse_args()

    try:
        if args.verify:
            success = cmd_verify()
            sys.exit(0 if success else 1)
        elif args.info:
            cmd_info()
            sys.exit(0)
        elif args.drop:
            confirm = input("Are you sure you want to drop all collections? (yes/no): ")
            if confirm.lower() == "yes":
                cmd_drop(args.collection)
            else:
                print("Aborted.")
            sys.exit(0)
        else:
            # Default: initialize
            success = cmd_init(
                recreate=args.recreate,
                skip_legacy=not args.include_legacy,
            )
            sys.exit(0 if success else 1)

    except ConnectionError as e:
        logger.error(f"Could not connect to Qdrant: {e}")
        logger.error("Make sure Qdrant is running or QDRANT_URL is configured correctly.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
