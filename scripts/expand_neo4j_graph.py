#!/usr/bin/env python3
"""
Neo4j Graph Expansion Script for AutoCognitix

Expands the diagnostic graph with Component and Repair nodes:
- 500+ Component nodes for vehicle parts
- 300+ Repair nodes with costs, difficulty, tools, time estimates
- 2000+ relationships (DTC->Component, Component->Repair)

Usage:
    python scripts/expand_neo4j_graph.py --components   # Add component nodes
    python scripts/expand_neo4j_graph.py --repairs      # Add repair nodes
    python scripts/expand_neo4j_graph.py --relationships # Create relationships
    python scripts/expand_neo4j_graph.py --all          # Full expansion
    python scripts/expand_neo4j_graph.py --stats        # Show statistics
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from neo4j import GraphDatabase
from tqdm import tqdm

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

# Data file paths
DATA_DIR = PROJECT_ROOT / "data" / "graph_expansion"
COMPONENTS_FILE = DATA_DIR / "components.json"
REPAIRS_FILE = DATA_DIR / "repairs.json"
DTC_COMPONENT_MAPPINGS_FILE = DATA_DIR / "dtc_component_mappings.json"
COMPONENT_REPAIR_MAPPINGS_FILE = DATA_DIR / "component_repair_mappings.json"

# Batch size for Neo4j operations
BATCH_SIZE = 100


def sanitize_string(value: Optional[str]) -> Optional[str]:
    """Sanitize string to prevent injection attacks."""
    if value is None:
        return None
    # Remove any potential Cypher injection characters
    value = str(value)
    # Allow alphanumeric, spaces, common punctuation, Hungarian characters
    # Remove backticks, semicolons, and other dangerous characters
    sanitized = re.sub(r'[`;\[\]{}|\\]', '', value)
    return sanitized.strip()


def sanitize_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively sanitize all string values in a dictionary."""
    sanitized = {}
    for key, value in data.items():
        if isinstance(value, str):
            sanitized[key] = sanitize_string(value)
        elif isinstance(value, dict):
            sanitized[key] = sanitize_dict(value)
        elif isinstance(value, list):
            sanitized[key] = [
                sanitize_string(v) if isinstance(v, str) else v
                for v in value
            ]
        else:
            sanitized[key] = value
    return sanitized


class Neo4jGraphExpander:
    """Handles Neo4j graph expansion with batch operations."""

    def __init__(self):
        """Initialize Neo4j connection."""
        self.uri = settings.NEO4J_URI
        self.user = settings.NEO4J_USER
        self.password = settings.NEO4J_PASSWORD
        self.driver = None

    def connect(self) -> bool:
        """Establish connection to Neo4j."""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info(f"Connected to Neo4j at {self.uri}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            return False

    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()

    def get_statistics(self) -> Dict[str, int]:
        """Get current graph statistics."""
        stats = {}
        with self.driver.session() as session:
            # Count nodes by label
            for label in ["DTC", "Symptom", "Component", "Repair", "Part"]:
                result = session.run(
                    f"MATCH (n:{label}) RETURN count(n) as count"
                )
                stats[f"{label.lower()}_nodes"] = result.single()["count"]

            # Count relationships
            for rel_type in ["CAUSES", "INDICATES_FAILURE_OF", "REPAIRED_BY", "USES_PART"]:
                result = session.run(
                    f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count"
                )
                stats[f"{rel_type.lower()}_rels"] = result.single()["count"]

        return stats

    def create_indexes(self):
        """Create indexes for better performance."""
        indexes = [
            "CREATE INDEX component_id IF NOT EXISTS FOR (c:Component) ON (c.id)",
            "CREATE INDEX component_name IF NOT EXISTS FOR (c:Component) ON (c.name)",
            "CREATE INDEX component_system IF NOT EXISTS FOR (c:Component) ON (c.system)",
            "CREATE INDEX repair_id IF NOT EXISTS FOR (r:Repair) ON (r.id)",
            "CREATE INDEX repair_name IF NOT EXISTS FOR (r:Repair) ON (r.name)",
            "CREATE INDEX repair_category IF NOT EXISTS FOR (r:Repair) ON (r.category)",
            "CREATE INDEX part_id IF NOT EXISTS FOR (p:Part) ON (p.id)",
        ]
        with self.driver.session() as session:
            for index_query in indexes:
                try:
                    session.run(index_query)
                except Exception as e:
                    logger.debug(f"Index may already exist: {e}")
        logger.info("Indexes created/verified")

    def load_components(self) -> List[Dict[str, Any]]:
        """Load components from JSON file."""
        if not COMPONENTS_FILE.exists():
            logger.error(f"Components file not found: {COMPONENTS_FILE}")
            return []
        with open(COMPONENTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("components", [])

    def load_repairs(self) -> List[Dict[str, Any]]:
        """Load repairs from JSON file."""
        if not REPAIRS_FILE.exists():
            logger.error(f"Repairs file not found: {REPAIRS_FILE}")
            return []
        with open(REPAIRS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("repairs", [])

    def load_dtc_component_mappings(self) -> List[Dict[str, Any]]:
        """Load DTC to Component mappings from JSON file."""
        if not DTC_COMPONENT_MAPPINGS_FILE.exists():
            logger.error(f"DTC-Component mappings file not found: {DTC_COMPONENT_MAPPINGS_FILE}")
            return []
        with open(DTC_COMPONENT_MAPPINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("mappings", [])

    def load_component_repair_mappings(self) -> List[Dict[str, Any]]:
        """Load Component to Repair mappings from JSON file."""
        if not COMPONENT_REPAIR_MAPPINGS_FILE.exists():
            logger.error(f"Component-Repair mappings file not found: {COMPONENT_REPAIR_MAPPINGS_FILE}")
            return []
        with open(COMPONENT_REPAIR_MAPPINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("mappings", [])

    def seed_components(self) -> int:
        """Seed Component nodes into Neo4j using batch operations."""
        components = self.load_components()
        if not components:
            return 0

        count = 0
        batch = []

        for comp in tqdm(components, desc="Preparing Component nodes"):
            comp_data = sanitize_dict(comp)
            batch.append(comp_data)

            if len(batch) >= BATCH_SIZE:
                count += self._create_component_batch(batch)
                batch = []

        # Process remaining
        if batch:
            count += self._create_component_batch(batch)

        logger.info(f"Created {count} Component nodes")
        return count

    def _create_component_batch(self, batch: List[Dict[str, Any]]) -> int:
        """Create a batch of Component nodes."""
        query = """
        UNWIND $batch AS comp
        MERGE (c:Component {id: comp.id})
        ON CREATE SET
            c.name = comp.name,
            c.name_hu = comp.name_hu,
            c.system = comp.system,
            c.subsystem = comp.subsystem,
            c.criticality = comp.criticality,
            c.created_at = datetime()
        ON MATCH SET
            c.name = comp.name,
            c.name_hu = comp.name_hu,
            c.system = comp.system,
            c.subsystem = comp.subsystem,
            c.criticality = comp.criticality,
            c.updated_at = datetime()
        RETURN count(c) as count
        """
        with self.driver.session() as session:
            result = session.run(query, batch=batch)
            return result.single()["count"]

    def seed_repairs(self) -> int:
        """Seed Repair nodes into Neo4j using batch operations."""
        repairs = self.load_repairs()
        if not repairs:
            return 0

        count = 0
        batch = []

        for repair in tqdm(repairs, desc="Preparing Repair nodes"):
            repair_data = sanitize_dict(repair)
            batch.append(repair_data)

            if len(batch) >= BATCH_SIZE:
                count += self._create_repair_batch(batch)
                batch = []

        # Process remaining
        if batch:
            count += self._create_repair_batch(batch)

        logger.info(f"Created {count} Repair nodes")
        return count

    def _create_repair_batch(self, batch: List[Dict[str, Any]]) -> int:
        """Create a batch of Repair nodes."""
        query = """
        UNWIND $batch AS repair
        MERGE (r:Repair {id: repair.id})
        ON CREATE SET
            r.name = repair.name,
            r.name_hu = repair.name_hu,
            r.difficulty = repair.difficulty,
            r.time_minutes = repair.time_minutes,
            r.cost_min = repair.cost_min,
            r.cost_max = repair.cost_max,
            r.tools = repair.tools,
            r.category = repair.category,
            r.created_at = datetime()
        ON MATCH SET
            r.name = repair.name,
            r.name_hu = repair.name_hu,
            r.difficulty = repair.difficulty,
            r.time_minutes = repair.time_minutes,
            r.cost_min = repair.cost_min,
            r.cost_max = repair.cost_max,
            r.tools = repair.tools,
            r.category = repair.category,
            r.updated_at = datetime()
        RETURN count(r) as count
        """
        with self.driver.session() as session:
            result = session.run(query, batch=batch)
            return result.single()["count"]

    def create_dtc_component_relationships(self) -> int:
        """Create relationships between DTC and Component nodes."""
        mappings = self.load_dtc_component_mappings()
        if not mappings:
            return 0

        count = 0

        # Get all DTC codes from database
        with self.driver.session() as session:
            result = session.run("MATCH (d:DTC) RETURN d.code as code")
            dtc_codes = {record["code"] for record in result}

        logger.info(f"Found {len(dtc_codes)} DTC nodes in database")

        batch = []
        for mapping in tqdm(mappings, desc="Preparing DTC-Component relationships"):
            dtc_pattern = mapping.get("dtc_pattern")
            components = mapping.get("components", [])
            failure_mode = sanitize_string(mapping.get("failure_mode", "malfunction"))
            confidence = mapping.get("confidence", 0.8)

            # Find matching DTC codes
            matching_dtcs = [
                code for code in dtc_codes
                if code.startswith(dtc_pattern)
            ]

            for dtc_code in matching_dtcs:
                for comp_id in components:
                    batch.append({
                        "dtc_code": dtc_code,
                        "comp_id": sanitize_string(comp_id),
                        "failure_mode": failure_mode,
                        "confidence": confidence
                    })

                    if len(batch) >= BATCH_SIZE:
                        count += self._create_dtc_component_batch(batch)
                        batch = []

        # Process remaining
        if batch:
            count += self._create_dtc_component_batch(batch)

        logger.info(f"Created {count} DTC-Component relationships")
        return count

    def _create_dtc_component_batch(self, batch: List[Dict[str, Any]]) -> int:
        """Create a batch of DTC-Component relationships."""
        query = """
        UNWIND $batch AS rel
        MATCH (d:DTC {code: rel.dtc_code})
        MATCH (c:Component {id: rel.comp_id})
        MERGE (d)-[r:INDICATES_FAILURE_OF]->(c)
        ON CREATE SET
            r.failure_mode = rel.failure_mode,
            r.confidence = rel.confidence,
            r.created_at = datetime()
        ON MATCH SET
            r.failure_mode = rel.failure_mode,
            r.confidence = rel.confidence,
            r.updated_at = datetime()
        RETURN count(r) as count
        """
        with self.driver.session() as session:
            result = session.run(query, batch=batch)
            return result.single()["count"]

    def create_component_repair_relationships(self) -> int:
        """Create relationships between Component and Repair nodes."""
        mappings = self.load_component_repair_mappings()
        if not mappings:
            return 0

        count = 0
        batch = []

        for mapping in tqdm(mappings, desc="Preparing Component-Repair relationships"):
            comp_id = sanitize_string(mapping.get("component"))
            repairs = mapping.get("repairs", [])
            primary_repair = sanitize_string(mapping.get("primary_repair"))

            for repair_id in repairs:
                is_primary = repair_id == primary_repair
                batch.append({
                    "comp_id": comp_id,
                    "repair_id": sanitize_string(repair_id),
                    "is_primary": is_primary
                })

                if len(batch) >= BATCH_SIZE:
                    count += self._create_component_repair_batch(batch)
                    batch = []

        # Process remaining
        if batch:
            count += self._create_component_repair_batch(batch)

        logger.info(f"Created {count} Component-Repair relationships")
        return count

    def _create_component_repair_batch(self, batch: List[Dict[str, Any]]) -> int:
        """Create a batch of Component-Repair relationships."""
        query = """
        UNWIND $batch AS rel
        MATCH (c:Component {id: rel.comp_id})
        MATCH (r:Repair {id: rel.repair_id})
        MERGE (c)-[rep:REPAIRED_BY]->(r)
        ON CREATE SET
            rep.is_primary = rel.is_primary,
            rep.created_at = datetime()
        ON MATCH SET
            rep.is_primary = rel.is_primary,
            rep.updated_at = datetime()
        RETURN count(rep) as count
        """
        with self.driver.session() as session:
            result = session.run(query, batch=batch)
            return result.single()["count"]

    def validate_graph(self) -> Dict[str, Any]:
        """Validate the expanded graph for consistency."""
        validation = {
            "orphan_components": 0,
            "orphan_repairs": 0,
            "components_without_repairs": 0,
            "dtc_coverage": 0,
            "issues": []
        }

        with self.driver.session() as session:
            # Check for orphan components (no incoming relationships)
            result = session.run("""
                MATCH (c:Component)
                WHERE NOT ()-[:INDICATES_FAILURE_OF]->(c)
                RETURN count(c) as count
            """)
            validation["orphan_components"] = result.single()["count"]

            # Check for components without repairs
            result = session.run("""
                MATCH (c:Component)
                WHERE NOT (c)-[:REPAIRED_BY]->()
                RETURN count(c) as count
            """)
            validation["components_without_repairs"] = result.single()["count"]

            # Check for orphan repairs
            result = session.run("""
                MATCH (r:Repair)
                WHERE NOT ()-[:REPAIRED_BY]->(r)
                RETURN count(r) as count
            """)
            validation["orphan_repairs"] = result.single()["count"]

            # Calculate DTC coverage
            result = session.run("""
                MATCH (d:DTC)
                WITH count(d) as total_dtc
                MATCH (d:DTC)-[:INDICATES_FAILURE_OF]->()
                WITH total_dtc, count(DISTINCT d) as connected_dtc
                RETURN
                    total_dtc,
                    connected_dtc,
                    toFloat(connected_dtc) / total_dtc * 100 as coverage_pct
            """)
            record = result.single()
            if record:
                validation["dtc_coverage"] = round(record["coverage_pct"], 2)
                validation["total_dtc"] = record["total_dtc"]
                validation["connected_dtc"] = record["connected_dtc"]

        return validation


def print_statistics(stats: Dict[str, int]):
    """Print graph statistics in a formatted way."""
    print("\n" + "=" * 50)
    print("Neo4j Graph Statistics")
    print("=" * 50)
    print("\nNodes:")
    print(f"  DTC nodes:       {stats.get('dtc_nodes', 0):,}")
    print(f"  Symptom nodes:   {stats.get('symptom_nodes', 0):,}")
    print(f"  Component nodes: {stats.get('component_nodes', 0):,}")
    print(f"  Repair nodes:    {stats.get('repair_nodes', 0):,}")
    print(f"  Part nodes:      {stats.get('part_nodes', 0):,}")
    print("\nRelationships:")
    print(f"  CAUSES:              {stats.get('causes_rels', 0):,}")
    print(f"  INDICATES_FAILURE_OF: {stats.get('indicates_failure_of_rels', 0):,}")
    print(f"  REPAIRED_BY:         {stats.get('repaired_by_rels', 0):,}")
    print(f"  USES_PART:           {stats.get('uses_part_rels', 0):,}")
    print("=" * 50 + "\n")


def print_validation(validation: Dict[str, Any]):
    """Print validation results."""
    print("\n" + "=" * 50)
    print("Graph Validation Results")
    print("=" * 50)
    print(f"\nDTC Coverage: {validation.get('dtc_coverage', 0)}%")
    print(f"  ({validation.get('connected_dtc', 0):,} / {validation.get('total_dtc', 0):,} DTCs linked to components)")
    print(f"\nOrphan Components: {validation.get('orphan_components', 0)}")
    print(f"Components without Repairs: {validation.get('components_without_repairs', 0)}")
    print(f"Orphan Repairs: {validation.get('orphan_repairs', 0)}")
    if validation.get("issues"):
        print("\nIssues found:")
        for issue in validation["issues"]:
            print(f"  - {issue}")
    print("=" * 50 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Expand Neo4j diagnostic graph with Component and Repair nodes"
    )
    parser.add_argument(
        "--components",
        action="store_true",
        help="Seed Component nodes only",
    )
    parser.add_argument(
        "--repairs",
        action="store_true",
        help="Seed Repair nodes only",
    )
    parser.add_argument(
        "--relationships",
        action="store_true",
        help="Create relationships only",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Full expansion (components, repairs, relationships)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show graph statistics",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate graph consistency",
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

    # Default to --all if no specific option is selected
    if not any([args.components, args.repairs, args.relationships, args.all, args.stats, args.validate]):
        args.all = True

    # Initialize expander
    expander = Neo4jGraphExpander()

    if not expander.connect():
        logger.error("Failed to connect to Neo4j. Exiting.")
        sys.exit(1)

    try:
        if args.stats:
            stats = expander.get_statistics()
            print_statistics(stats)
            return

        if args.validate:
            validation = expander.validate_graph()
            print_validation(validation)
            return

        # Show initial stats
        logger.info("Getting initial graph statistics...")
        initial_stats = expander.get_statistics()
        print_statistics(initial_stats)

        # Create indexes
        expander.create_indexes()

        if args.components or args.all:
            logger.info("Seeding Component nodes...")
            comp_count = expander.seed_components()

        if args.repairs or args.all:
            logger.info("Seeding Repair nodes...")
            repair_count = expander.seed_repairs()

        if args.relationships or args.all:
            logger.info("Creating DTC-Component relationships...")
            dtc_comp_count = expander.create_dtc_component_relationships()

            logger.info("Creating Component-Repair relationships...")
            comp_repair_count = expander.create_component_repair_relationships()

        # Show final stats
        logger.info("Getting final graph statistics...")
        final_stats = expander.get_statistics()
        print_statistics(final_stats)

        # Validate
        logger.info("Validating graph...")
        validation = expander.validate_graph()
        print_validation(validation)

        logger.info("Graph expansion completed successfully!")

    except Exception as e:
        logger.error(f"Error during graph expansion: {e}")
        raise
    finally:
        expander.close()


if __name__ == "__main__":
    main()
