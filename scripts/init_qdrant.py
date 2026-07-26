#!/usr/bin/env python3
"""
Qdrant Vector Database Initialization Script for AutoCognitix.

This script initializes all required Qdrant collections with proper
configuration for the AutoCognitix diagnostic system.

Usage:
    python scripts/init_qdrant.py                    # Initialize collections
    python scripts/init_qdrant.py --verify           # Verify collections exist
    python scripts/init_qdrant.py --info             # Show collection info
    python scripts/init_qdrant.py --recreate         # Drop and recreate all
    python scripts/init_qdrant.py --drop             # Drop all collections
    python scripts/init_qdrant.py --drop --collection X   # Drop just X

Destructive operations refuse any collection that still holds points. Add
--force to override, and read it as "for every collection this command touches",
not just one. See assert_safe_to_delete().

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


def assert_safe_to_delete(client: QdrantClient, name: str, force: bool = False) -> None:
    """Refuse to delete a collection that still holds points, unless forced.

    The guard is on the POINT COUNT, not on a list of protected names. A name
    list would have to be maintained by whoever adds the next collection, and
    would be wrong the moment it was not - whereas "this collection has data in
    it" is true or false at the moment of the call and needs no upkeep.

    This exists because these names are not empty in production the way the code
    assumes: ``dtc_embeddings_hu`` holds 2,323 points and
    ``symptom_embeddings_hu`` 117, written by the indexer scripts. They are a
    partial stale copy that the application no longer reads - but the decision to
    destroy them belongs to a human who has seen the count, not to a flag that
    happens to be on.

    Note this covers ``--drop --collection <name>`` too, which takes a free-form
    name and never consults ``COLLECTIONS`` - so it can target the unified
    ``autocognitix`` store holding every vector the application actually reads.
    Nothing about the COLLECTIONS list bounds the blast radius; this function is
    the only thing that does.

    Raises:
        RuntimeError: if the collection holds points and ``force`` is not set,
            or if the point count cannot be read at all.
    """
    try:
        count = client.count(collection_name=name, exact=True).count
    except Exception as e:
        # Fail CLOSED, and note --force does NOT open this branch: "destroy it
        # anyway" is a decision about a known quantity, and here the quantity is
        # unknown. Deliberately does not tell the operator to retry with --force,
        # because that produces a byte-identical refusal - an instruction that
        # loops the reader is worse than none.
        raise RuntimeError(
            f"Refusing to delete '{name}': could not read its point count ({e}). "
            f"Fix the connection or the API key's permissions so the count can be "
            f"read, or delete the collection from the Qdrant console if you have "
            f"confirmed there what it holds. --force does not bypass this."
        ) from e

    # `count is None` is treated as unknown, not as zero. CountResult.count is a
    # required int so a real server cannot produce it, but a bare `if count`
    # would silently read None as "empty" and unlock the delete - the one
    # direction this function must never fail in.
    if count is None:
        raise RuntimeError(
            f"Refusing to delete '{name}': the server returned no point count. "
            f"--force does not bypass this."
        )

    if count > 0 and not force:
        raise RuntimeError(
            f"Refusing to delete '{name}': it holds {count:,} points. "
            f"Nothing in the application reads this collection, but the data is "
            f"real and its removal is not reversible. Re-run with --force if you "
            f"have decided to destroy it."
        )
    if count > 0:
        logger.warning(f"--force given: destroying {count:,} points in '{name}'")


def create_collection(
    client: QdrantClient,
    config: Dict[str, Any],
    recreate: bool = False,
    force: bool = False,
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
                # --recreate had NO confirmation of any kind, unlike --drop which
                # at least prompts. It was the sharper of the two edges.
                assert_safe_to_delete(client, name, force=force)
                logger.info(f"Deleting existing collection: {name}")
                client.delete_collection(collection_name=name)
            else:
                logger.info(f"Collection already exists: {name}")
                return True

        # Create the collection
        logger.info(
            f"Creating collection: {name} (vectors: {vector_size}-dim, distance: {distance})"
        )

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

    except RuntimeError:
        # The deletion guard refused. Do NOT let it decay into `return False`:
        # the caller treats False as "this one collection had a problem, carry
        # on with the rest", which turns a deliberate refusal to destroy data
        # into a line in a tally. Abort the run so the operator sees it.
        raise
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
                "vector_size": info.config.params.vectors.size
                if hasattr(info.config.params.vectors, "size")
                else "multi",
                "distance": str(info.config.params.vectors.distance)
                if hasattr(info.config.params.vectors, "distance")
                else "unknown",
            },
        }
    except Exception:
        return {"name": name, "status": "not_found"}


# =============================================================================
# CLI Commands
# =============================================================================


def cmd_init(recreate: bool = False, skip_legacy: bool = True, force: bool = False) -> bool:
    """Initialize all collections.

    With ``recreate``, every candidate is checked BEFORE any is deleted, for the
    same reason ``cmd_drop`` does it: a run that deletes and recreates two
    collections and then refuses on the third leaves a state nobody chose.

    Without the pre-flight this was safe only by list ORDERING - the collection
    holding the most points happens to sit first in ``COLLECTIONS``, so it
    refused before anything was touched. That is luck, and it inverts the moment
    someone reorders the list or empties that collection.
    """
    client = get_qdrant_client()

    targets = [c for c in COLLECTIONS if not (c.get("legacy") and skip_legacy)]

    if recreate:
        for config in targets:
            if collection_exists(client, config["name"]):
                assert_safe_to_delete(client, config["name"], force=force)

    success_count = 0
    total_count = 0

    for config in COLLECTIONS:
        # Skip legacy collections by default
        if config.get("legacy") and skip_legacy:
            logger.info(f"Skipping legacy collection: {config['name']}")
            continue

        total_count += 1
        if create_collection(client, config, recreate=recreate, force=force):
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


def cmd_drop(collection_name: Optional[str] = None, force: bool = False) -> bool:
    """Drop one or all collections.

    Every deletion goes through :func:`assert_safe_to_delete` first, so a
    collection that still holds points is refused unless ``--force`` is given.

    ``collection_name`` is a free-form name resolved against the LIVE server, not
    against ``COLLECTIONS`` - so this can target the unified ``autocognitix``
    store. That is exactly why the guard counts points instead of checking a
    name list.
    """
    client = get_qdrant_client()

    # Read the live list ONCE and let failures propagate. Going through
    # collection_exists() per name swallows the error and returns False, which
    # turns "Qdrant is unreachable" into "there was nothing to drop" - reported
    # as success, with exit 0. The error has to stay distinguishable from a
    # genuinely empty result.
    existing = {c.name for c in client.get_collections().collections}

    if collection_name:
        # Drop specific collection
        if collection_name in existing:
            assert_safe_to_delete(client, collection_name, force=force)
            client.delete_collection(collection_name=collection_name)
            logger.info(f"Dropped collection: {collection_name}")
            return True
        else:
            logger.warning(f"Collection not found: {collection_name}")
            return False
    else:
        # Drop all collections. Every candidate is checked BEFORE anything is
        # deleted: a partial drop that stops halfway through leaves the store in
        # a state nobody chose, which is worse than refusing outright.
        present = [c["name"] for c in COLLECTIONS if collection_exists(client, c["name"])]
        for name in present:
            assert_safe_to_delete(client, name, force=force)

        for name in present:
            client.delete_collection(collection_name=name)
            logger.info(f"Dropped collection: {name}")

        logger.info(f"Dropped {len(present)} collection(s)")
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
    python scripts/init_qdrant.py --info       # Show collection details
    python scripts/init_qdrant.py --recreate   # Drop and recreate all
    python scripts/init_qdrant.py --drop       # Drop all collections
    python scripts/init_qdrant.py --drop --collection dtc_embeddings_hu

--recreate and --drop REFUSE any collection that still holds points. Add --force
to override - it applies to every collection the command touches, not one.
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
        help=(
            "Drop and recreate ALL non-legacy collections. Refused for any that "
            "still holds points unless --force is also given. Cannot be scoped "
            "with --collection."
        ),
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show detailed collection information",
    )
    parser.add_argument(
        "--drop",
        action="store_true",
        help=(
            "Drop all collections, or just --collection if given (dangerous!). "
            "Refused for any that still holds points unless --force is given."
        ),
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
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Allow --drop/--recreate to destroy collections that still hold "
            "points. Without it, any non-empty collection is refused. NOTE this "
            "applies to EVERY collection the command touches, not just one."
        ),
    )

    args = parser.parse_args()

    # --recreate loops all non-legacy COLLECTIONS and has never read
    # --collection. Silently ignoring it was survivable while every deletion was
    # unconditional; combined with --force it is not, because the operator reads
    # "--recreate --collection X --force" as scoping the destruction to X and
    # gets all of them. Refuse the combination rather than honour half of it.
    if args.recreate and args.collection:
        parser.error(
            "--collection cannot be combined with --recreate: --recreate always "
            "operates on every non-legacy collection. Use --drop --collection "
            "<name> to act on a single collection."
        )

    try:
        if args.verify:
            success = cmd_verify()
            sys.exit(0 if success else 1)
        elif args.info:
            cmd_info()
            sys.exit(0)
        elif args.drop:
            scope = f"collection '{args.collection}'" if args.collection else "ALL collections"
            confirm = input(f"Are you sure you want to drop {scope}? (yes/no): ")
            if confirm.lower() != "yes":
                print("Aborted.")
                sys.exit(0)
            # The return value used to be discarded, so dropping a collection
            # that was not there reported success.
            sys.exit(0 if cmd_drop(args.collection, force=args.force) else 1)
        else:
            # Default: initialize
            success = cmd_init(
                recreate=args.recreate,
                skip_legacy=not args.include_legacy,
                force=args.force,
            )
            sys.exit(0 if success else 1)

    except ConnectionError as e:
        logger.error(f"Could not connect to Qdrant: {e}")
        logger.error("Make sure Qdrant is running or QDRANT_URL is configured correctly.")
        sys.exit(1)
    except RuntimeError as e:
        # A refusal by assert_safe_to_delete. It is a decision, not a crash, so
        # it exits cleanly instead of dumping a traceback - a stack trace reads
        # as "the tool broke" to a human and to anything scraping stderr, and the
        # message already says exactly what happened and what to do about it.
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
