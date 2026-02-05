#!/usr/bin/env python3
"""
Neo4j Schema and Index Setup Script for AutoCognitix

This script creates necessary indexes and constraints for efficient
graph queries in the AutoCognitix diagnostic system.

Usage:
    python scripts/setup_neo4j_indexes.py
    python scripts/setup_neo4j_indexes.py --verify  # Verify existing indexes
    python scripts/setup_neo4j_indexes.py --drop    # Drop and recreate indexes

Indexes are critical for:
- Fast DTC code lookup by code (unique constraint)
- Symptom search by name for diagnostic matching
- Component lookup for repair path traversal
- Full-text search for Hungarian descriptions
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from neo4j import GraphDatabase

from backend.app.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Index definitions: (label, property, index_type, is_unique)
# index_type: "btree" for range queries, "text" for full-text search
INDEXES: List[Tuple[str, str, str, bool]] = [
    # Primary lookup indexes (unique constraints)
    ("DTCNode", "code", "btree", True),

    # Regular indexes for fast lookups
    ("SymptomNode", "name", "btree", False),
    ("ComponentNode", "name", "btree", False),
    ("RepairNode", "name", "btree", False),
    ("PartNode", "name", "btree", False),
    ("PartNode", "part_number", "btree", False),
    ("TestPointNode", "name", "btree", False),
    ("VehicleNode", "make", "btree", False),
    ("VehicleNode", "model", "btree", False),

    # Category/system indexes for filtering
    ("DTCNode", "category", "btree", False),
    ("DTCNode", "severity", "btree", False),
    ("ComponentNode", "system", "btree", False),
]

# Full-text indexes for Hungarian description search
FULLTEXT_INDEXES: List[Tuple[str, str, List[str]]] = [
    # (index_name, label, properties)
    ("dtc_description_hu_idx", "DTCNode", ["description_hu", "description_en"]),
    ("symptom_description_hu_idx", "SymptomNode", ["description", "description_hu"]),
    ("component_name_hu_idx", "ComponentNode", ["name", "name_hu"]),
    ("repair_description_hu_idx", "RepairNode", ["name", "description_hu"]),
]

# Composite indexes for common query patterns
COMPOSITE_INDEXES: List[Tuple[str, str, List[str]]] = [
    # (index_name, label, properties)
    ("vehicle_make_model_idx", "VehicleNode", ["make", "model"]),
]


def get_neo4j_driver():
    """Create Neo4j driver from settings."""
    uri = settings.NEO4J_URI
    user = settings.NEO4J_USER
    password = settings.NEO4J_PASSWORD

    logger.info(f"Connecting to Neo4j at {uri}")
    return GraphDatabase.driver(uri, auth=(user, password))


def get_existing_indexes(driver) -> dict:
    """Get all existing indexes from Neo4j."""
    with driver.session() as session:
        result = session.run("SHOW INDEXES")
        indexes = {}
        for record in result:
            name = record.get("name", "")
            label = record.get("labelsOrTypes", [])
            props = record.get("properties", [])
            index_type = record.get("type", "")
            state = record.get("state", "")
            indexes[name] = {
                "labels": label,
                "properties": props,
                "type": index_type,
                "state": state,
            }
        return indexes


def get_existing_constraints(driver) -> dict:
    """Get all existing constraints from Neo4j."""
    with driver.session() as session:
        result = session.run("SHOW CONSTRAINTS")
        constraints = {}
        for record in result:
            name = record.get("name", "")
            label = record.get("labelsOrTypes", [])
            props = record.get("properties", [])
            constraint_type = record.get("type", "")
            constraints[name] = {
                "labels": label,
                "properties": props,
                "type": constraint_type,
            }
        return constraints


def create_index(driver, label: str, property_name: str, is_unique: bool = False) -> bool:
    """Create a single property index or unique constraint."""
    index_name = f"{label.lower()}_{property_name}_idx"

    with driver.session() as session:
        try:
            if is_unique:
                # Create unique constraint (which includes an index)
                constraint_name = f"{label.lower()}_{property_name}_unique"
                query = f"""
                CREATE CONSTRAINT {constraint_name} IF NOT EXISTS
                FOR (n:{label})
                REQUIRE n.{property_name} IS UNIQUE
                """
                session.run(query)
                logger.info(f"Created unique constraint: {constraint_name} on {label}.{property_name}")
            else:
                # Create regular index
                query = f"""
                CREATE INDEX {index_name} IF NOT EXISTS
                FOR (n:{label})
                ON (n.{property_name})
                """
                session.run(query)
                logger.info(f"Created index: {index_name} on {label}.{property_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create index {index_name}: {e}")
            return False


def create_fulltext_index(driver, index_name: str, label: str, properties: List[str]) -> bool:
    """Create a full-text index for search functionality."""
    props_str = ", ".join([f"n.{p}" for p in properties])

    with driver.session() as session:
        try:
            # Drop existing index if exists (full-text indexes don't support IF NOT EXISTS in older versions)
            try:
                session.run(f"DROP INDEX {index_name} IF EXISTS")
            except Exception:
                pass

            query = f"""
            CREATE FULLTEXT INDEX {index_name}
            FOR (n:{label})
            ON EACH [{props_str}]
            """
            session.run(query)
            logger.info(f"Created fulltext index: {index_name} on {label}[{', '.join(properties)}]")
            return True
        except Exception as e:
            logger.error(f"Failed to create fulltext index {index_name}: {e}")
            return False


def create_composite_index(driver, index_name: str, label: str, properties: List[str]) -> bool:
    """Create a composite index on multiple properties."""
    props_str = ", ".join([f"n.{p}" for p in properties])

    with driver.session() as session:
        try:
            query = f"""
            CREATE INDEX {index_name} IF NOT EXISTS
            FOR (n:{label})
            ON ({props_str})
            """
            session.run(query)
            logger.info(f"Created composite index: {index_name} on {label}[{', '.join(properties)}]")
            return True
        except Exception as e:
            logger.error(f"Failed to create composite index {index_name}: {e}")
            return False


def drop_all_custom_indexes(driver) -> int:
    """Drop all custom indexes (not system indexes)."""
    count = 0

    # Get existing indexes
    indexes = get_existing_indexes(driver)
    constraints = get_existing_constraints(driver)

    with driver.session() as session:
        # Drop constraints first
        for name in constraints.keys():
            if not name.startswith("constraint_"):  # Skip system constraints
                try:
                    session.run(f"DROP CONSTRAINT {name} IF EXISTS")
                    logger.info(f"Dropped constraint: {name}")
                    count += 1
                except Exception as e:
                    logger.warning(f"Could not drop constraint {name}: {e}")

        # Drop indexes
        for name in indexes.keys():
            if not name.startswith("index_"):  # Skip system indexes
                try:
                    session.run(f"DROP INDEX {name} IF EXISTS")
                    logger.info(f"Dropped index: {name}")
                    count += 1
                except Exception as e:
                    logger.warning(f"Could not drop index {name}: {e}")

    return count


def verify_indexes(driver) -> Tuple[int, int]:
    """Verify all expected indexes exist and are online."""
    indexes = get_existing_indexes(driver)
    constraints = get_existing_constraints(driver)

    expected = 0
    found = 0

    logger.info("\n=== Index Verification Report ===\n")

    # Check regular indexes
    for label, prop, _, is_unique in INDEXES:
        expected += 1
        index_name = f"{label.lower()}_{prop}_idx"
        constraint_name = f"{label.lower()}_{prop}_unique"

        if is_unique:
            if constraint_name in constraints:
                found += 1
                logger.info(f"[OK] Unique constraint: {constraint_name}")
            else:
                logger.warning(f"[MISSING] Unique constraint: {constraint_name} on {label}.{prop}")
        else:
            if index_name in indexes:
                state = indexes[index_name].get("state", "UNKNOWN")
                if state == "ONLINE":
                    found += 1
                    logger.info(f"[OK] Index: {index_name} (ONLINE)")
                else:
                    logger.warning(f"[PENDING] Index: {index_name} (state: {state})")
            else:
                logger.warning(f"[MISSING] Index: {index_name} on {label}.{prop}")

    # Check fulltext indexes
    for index_name, label, props in FULLTEXT_INDEXES:
        expected += 1
        if index_name in indexes:
            found += 1
            logger.info(f"[OK] Fulltext index: {index_name}")
        else:
            logger.warning(f"[MISSING] Fulltext index: {index_name} on {label}")

    # Check composite indexes
    for index_name, label, props in COMPOSITE_INDEXES:
        expected += 1
        if index_name in indexes:
            found += 1
            logger.info(f"[OK] Composite index: {index_name}")
        else:
            logger.warning(f"[MISSING] Composite index: {index_name} on {label}")

    logger.info(f"\n=== Summary: {found}/{expected} indexes present ===\n")

    return found, expected


def setup_indexes(driver, drop_existing: bool = False) -> Tuple[int, int]:
    """Set up all indexes."""
    success = 0
    total = 0

    if drop_existing:
        dropped = drop_all_custom_indexes(driver)
        logger.info(f"Dropped {dropped} existing indexes/constraints")

    logger.info("\n=== Creating Indexes ===\n")

    # Create regular indexes
    for label, prop, index_type, is_unique in INDEXES:
        total += 1
        if create_index(driver, label, prop, is_unique):
            success += 1

    # Create fulltext indexes
    for index_name, label, props in FULLTEXT_INDEXES:
        total += 1
        if create_fulltext_index(driver, index_name, label, props):
            success += 1

    # Create composite indexes
    for index_name, label, props in COMPOSITE_INDEXES:
        total += 1
        if create_composite_index(driver, index_name, label, props):
            success += 1

    logger.info(f"\n=== Created {success}/{total} indexes ===\n")

    return success, total


def print_cypher_commands():
    """Print all Cypher commands for manual execution."""
    print("\n" + "=" * 60)
    print("Neo4j Cypher Commands for Manual Execution")
    print("=" * 60 + "\n")

    print("-- Unique Constraints --")
    for label, prop, _, is_unique in INDEXES:
        if is_unique:
            constraint_name = f"{label.lower()}_{prop}_unique"
            print(f"""
CREATE CONSTRAINT {constraint_name} IF NOT EXISTS
FOR (n:{label})
REQUIRE n.{prop} IS UNIQUE;
""")

    print("\n-- Regular Indexes --")
    for label, prop, _, is_unique in INDEXES:
        if not is_unique:
            index_name = f"{label.lower()}_{prop}_idx"
            print(f"""
CREATE INDEX {index_name} IF NOT EXISTS
FOR (n:{label})
ON (n.{prop});
""")

    print("\n-- Fulltext Indexes --")
    for index_name, label, props in FULLTEXT_INDEXES:
        props_str = ", ".join([f"n.{p}" for p in props])
        print(f"""
CREATE FULLTEXT INDEX {index_name}
FOR (n:{label})
ON EACH [{props_str}];
""")

    print("\n-- Composite Indexes --")
    for index_name, label, props in COMPOSITE_INDEXES:
        props_str = ", ".join([f"n.{p}" for p in props])
        print(f"""
CREATE INDEX {index_name} IF NOT EXISTS
FOR (n:{label})
ON ({props_str});
""")

    print("=" * 60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Set up Neo4j indexes for AutoCognitix"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify existing indexes without creating new ones",
    )
    parser.add_argument(
        "--drop",
        action="store_true",
        help="Drop existing indexes before creating new ones",
    )
    parser.add_argument(
        "--print-cypher",
        action="store_true",
        help="Print Cypher commands without executing them",
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

    if args.print_cypher:
        print_cypher_commands()
        return

    try:
        driver = get_neo4j_driver()

        # Test connection
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            result.single()
        logger.info("Successfully connected to Neo4j")

        if args.verify:
            found, expected = verify_indexes(driver)
            if found < expected:
                logger.warning("Some indexes are missing. Run without --verify to create them.")
                sys.exit(1)
        else:
            success, total = setup_indexes(driver, drop_existing=args.drop)

            # Verify after creation
            logger.info("\nVerifying created indexes...")
            verify_indexes(driver)

            if success < total:
                logger.warning("Some indexes failed to create. Check logs for details.")
                sys.exit(1)

        driver.close()
        logger.info("Neo4j index setup completed successfully!")

    except Exception as e:
        logger.error(f"Error during Neo4j setup: {e}")
        raise


if __name__ == "__main__":
    main()
