#!/usr/bin/env python3
"""
Neo4j Aura Seed Script for AutoCognitix

Seeds all diagnostic graph data to Neo4j Aura (cloud) from JSON data files.

Data sources:
- DTC codes from data/dtc_codes/all_codes_merged.json
- Components from data/graph_expansion/components.json
- Repairs from data/graph_expansion/repairs.json
- DTC-Component mappings from data/graph_expansion/dtc_component_mappings.json
- Component-Repair mappings from data/graph_expansion/component_repair_mappings.json
- Symptoms from data/symptoms/symptoms_hu.json

Environment variables required:
- NEO4J_URI: Neo4j Aura connection URI (e.g., neo4j+s://xxx.databases.neo4j.io)
- NEO4J_USER: Neo4j username (default: neo4j)
- NEO4J_PASSWORD: Neo4j password

Usage:
    python scripts/seed_neo4j_aura.py                    # Full seed
    python scripts/seed_neo4j_aura.py --dry-run         # Preview without changes
    python scripts/seed_neo4j_aura.py --nodes-only      # Seed only nodes
    python scripts/seed_neo4j_aura.py --relationships-only  # Seed only relationships
    python scripts/seed_neo4j_aura.py --clear           # Clear all data first
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Data file paths
DTC_CODES_PATH = DATA_DIR / "dtc_codes" / "all_codes_merged.json"
COMPONENTS_PATH = DATA_DIR / "graph_expansion" / "components.json"
REPAIRS_PATH = DATA_DIR / "graph_expansion" / "repairs.json"
DTC_COMPONENT_MAPPINGS_PATH = DATA_DIR / "graph_expansion" / "dtc_component_mappings.json"
COMPONENT_REPAIR_MAPPINGS_PATH = DATA_DIR / "graph_expansion" / "component_repair_mappings.json"
SYMPTOMS_PATH = DATA_DIR / "symptoms" / "symptoms_hu.json"

# Batch size for Neo4j operations
BATCH_SIZE = 500


@dataclass
class SeedStats:
    """Statistics for seeding operations."""

    dtc_nodes: int = 0
    component_nodes: int = 0
    repair_nodes: int = 0
    symptom_nodes: int = 0
    category_nodes: int = 0
    dtc_component_rels: int = 0
    component_repair_rels: int = 0
    dtc_symptom_rels: int = 0
    symptom_category_rels: int = 0

    def summary(self) -> str:
        """Return summary string."""
        return f"""
Seeding Summary:
================
Nodes Created:
  - DTC codes:    {self.dtc_nodes}
  - Components:   {self.component_nodes}
  - Repairs:      {self.repair_nodes}
  - Symptoms:     {self.symptom_nodes}
  - Categories:   {self.category_nodes}

Relationships Created:
  - DTC -> Component (INDICATES_FAILURE_OF): {self.dtc_component_rels}
  - Component -> Repair (REPAIRED_BY):       {self.component_repair_rels}
  - DTC -> Symptom (CAUSES):                 {self.dtc_symptom_rels}
  - Symptom -> Category (BELONGS_TO):        {self.symptom_category_rels}

Total Nodes: {self.dtc_nodes + self.component_nodes + self.repair_nodes + self.symptom_nodes + self.category_nodes}
Total Relationships: {self.dtc_component_rels + self.component_repair_rels + self.dtc_symptom_rels + self.symptom_category_rels}
"""


class Neo4jSeeder:
    """Handles seeding data to Neo4j Aura."""

    def __init__(self, uri: str, user: str, password: str, dry_run: bool = False):
        """Initialize the seeder with connection details."""
        self.uri = uri
        self.user = user
        self.password = password
        self.dry_run = dry_run
        self.driver = None
        self.stats = SeedStats()

    def connect(self) -> None:
        """Establish connection to Neo4j."""
        if self.dry_run:
            logger.info("[DRY-RUN] Would connect to Neo4j at %s", self.uri)
            return

        logger.info("Connecting to Neo4j at %s...", self.uri)
        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password),
            max_connection_lifetime=3600,
        )
        # Verify connection
        self.driver.verify_connectivity()
        logger.info("Successfully connected to Neo4j Aura")

    def close(self) -> None:
        """Close the connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

    def _run_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """Execute a Cypher query."""
        if self.dry_run:
            return []

        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    def _run_batch_query(
        self, query: str, data: List[Dict], batch_size: int = BATCH_SIZE, desc: str = ""
    ) -> int:
        """Execute a batch query with progress bar."""
        if self.dry_run:
            logger.info("[DRY-RUN] Would execute batch query for %d items: %s", len(data), desc)
            return len(data)

        total_processed = 0
        for i in tqdm(range(0, len(data), batch_size), desc=desc, unit="batch"):
            batch = data[i : i + batch_size]
            with self.driver.session() as session:
                result = session.run(query, {"batch": batch})
                summary = result.consume()
                total_processed += summary.counters.nodes_created + summary.counters.relationships_created

        return total_processed

    def clear_database(self) -> None:
        """Clear all nodes and relationships from the database."""
        if self.dry_run:
            logger.info("[DRY-RUN] Would clear all data from database")
            return

        logger.warning("Clearing all data from Neo4j database...")

        # Delete relationships first, then nodes
        queries = [
            "MATCH ()-[r]->() DELETE r",
            "MATCH (n) DELETE n",
        ]

        for query in queries:
            self._run_query(query)

        logger.info("Database cleared successfully")

    def create_constraints(self) -> None:
        """Create uniqueness constraints for better performance."""
        if self.dry_run:
            logger.info("[DRY-RUN] Would create uniqueness constraints")
            return

        logger.info("Creating uniqueness constraints...")

        constraints = [
            ("dtc_code_unique", "DTC", "code"),
            ("component_id_unique", "Component", "id"),
            ("repair_id_unique", "Repair", "id"),
            ("symptom_id_unique", "Symptom", "id"),
            ("category_id_unique", "Category", "id"),
        ]

        for name, label, prop in constraints:
            query = f"""
            CREATE CONSTRAINT {name} IF NOT EXISTS
            FOR (n:{label})
            REQUIRE n.{prop} IS UNIQUE
            """
            try:
                self._run_query(query)
                logger.debug("Created constraint: %s", name)
            except Exception as e:
                logger.warning("Constraint %s may already exist: %s", name, e)

        logger.info("Constraints created/verified")

    def create_indexes(self) -> None:
        """Create indexes for better query performance."""
        if self.dry_run:
            logger.info("[DRY-RUN] Would create indexes")
            return

        logger.info("Creating indexes...")

        indexes = [
            ("idx_dtc_category", "DTC", "category"),
            ("idx_dtc_severity", "DTC", "severity"),
            ("idx_component_system", "Component", "system"),
            ("idx_repair_difficulty", "Repair", "difficulty"),
            ("idx_symptom_severity", "Symptom", "severity"),
        ]

        for name, label, prop in indexes:
            query = f"""
            CREATE INDEX {name} IF NOT EXISTS
            FOR (n:{label})
            ON (n.{prop})
            """
            try:
                self._run_query(query)
                logger.debug("Created index: %s", name)
            except Exception as e:
                logger.warning("Index %s may already exist: %s", name, e)

        logger.info("Indexes created/verified")

    def seed_dtc_nodes(self, dtc_codes: List[Dict]) -> int:
        """Seed DTC code nodes."""
        logger.info("Seeding %d DTC nodes...", len(dtc_codes))

        query = """
        UNWIND $batch AS row
        MERGE (d:DTC {code: row.code})
        ON CREATE SET
            d.description_en = row.description_en,
            d.description_hu = row.description_hu,
            d.category = row.category,
            d.severity = row.severity,
            d.system = row.system,
            d.is_generic = row.is_generic,
            d.symptoms = row.symptoms,
            d.possible_causes = row.possible_causes,
            d.diagnostic_steps = row.diagnostic_steps,
            d.related_codes = row.related_codes,
            d.created_at = datetime()
        ON MATCH SET
            d.description_en = row.description_en,
            d.description_hu = row.description_hu,
            d.category = row.category,
            d.severity = row.severity,
            d.system = row.system,
            d.is_generic = row.is_generic,
            d.symptoms = row.symptoms,
            d.possible_causes = row.possible_causes,
            d.diagnostic_steps = row.diagnostic_steps,
            d.related_codes = row.related_codes,
            d.updated_at = datetime()
        """

        # Prepare data
        data = []
        for dtc in dtc_codes:
            data.append({
                "code": dtc["code"],
                "description_en": dtc.get("description_en", ""),
                "description_hu": dtc.get("description_hu"),
                "category": dtc.get("category", "unknown"),
                "severity": dtc.get("severity", "medium"),
                "system": dtc.get("system"),
                "is_generic": dtc.get("is_generic", True),
                "symptoms": dtc.get("symptoms", []),
                "possible_causes": dtc.get("possible_causes", []),
                "diagnostic_steps": dtc.get("diagnostic_steps", []),
                "related_codes": dtc.get("related_codes", []),
            })

        self._run_batch_query(query, data, desc="DTC nodes")
        self.stats.dtc_nodes = len(data)
        return len(data)

    def seed_component_nodes(self, components: List[Dict]) -> int:
        """Seed Component nodes."""
        logger.info("Seeding %d Component nodes...", len(components))

        query = """
        UNWIND $batch AS row
        MERGE (c:Component {id: row.id})
        ON CREATE SET
            c.name = row.name,
            c.name_hu = row.name_hu,
            c.system = row.system,
            c.subsystem = row.subsystem,
            c.criticality = row.criticality,
            c.created_at = datetime()
        ON MATCH SET
            c.name = row.name,
            c.name_hu = row.name_hu,
            c.system = row.system,
            c.subsystem = row.subsystem,
            c.criticality = row.criticality,
            c.updated_at = datetime()
        """

        data = []
        for comp in components:
            data.append({
                "id": comp["id"],
                "name": comp["name"],
                "name_hu": comp.get("name_hu"),
                "system": comp.get("system"),
                "subsystem": comp.get("subsystem"),
                "criticality": comp.get("criticality", "medium"),
            })

        self._run_batch_query(query, data, desc="Component nodes")
        self.stats.component_nodes = len(data)
        return len(data)

    def seed_repair_nodes(self, repairs: List[Dict]) -> int:
        """Seed Repair nodes."""
        logger.info("Seeding %d Repair nodes...", len(repairs))

        query = """
        UNWIND $batch AS row
        MERGE (r:Repair {id: row.id})
        ON CREATE SET
            r.name = row.name,
            r.name_hu = row.name_hu,
            r.difficulty = row.difficulty,
            r.time_minutes = row.time_minutes,
            r.cost_min = row.cost_min,
            r.cost_max = row.cost_max,
            r.tools = row.tools,
            r.category = row.category,
            r.created_at = datetime()
        ON MATCH SET
            r.name = row.name,
            r.name_hu = row.name_hu,
            r.difficulty = row.difficulty,
            r.time_minutes = row.time_minutes,
            r.cost_min = row.cost_min,
            r.cost_max = row.cost_max,
            r.tools = row.tools,
            r.category = row.category,
            r.updated_at = datetime()
        """

        data = []
        for repair in repairs:
            data.append({
                "id": repair["id"],
                "name": repair["name"],
                "name_hu": repair.get("name_hu"),
                "difficulty": repair.get("difficulty", "intermediate"),
                "time_minutes": repair.get("time_minutes"),
                "cost_min": repair.get("cost_min"),
                "cost_max": repair.get("cost_max"),
                "tools": repair.get("tools", []),
                "category": repair.get("category"),
            })

        self._run_batch_query(query, data, desc="Repair nodes")
        self.stats.repair_nodes = len(data)
        return len(data)

    def seed_symptom_nodes(self, symptoms: List[Dict]) -> int:
        """Seed Symptom nodes."""
        logger.info("Seeding %d Symptom nodes...", len(symptoms))

        query = """
        UNWIND $batch AS row
        MERGE (s:Symptom {id: row.id})
        ON CREATE SET
            s.name_hu = row.name_hu,
            s.name_en = row.name_en,
            s.description_hu = row.description_hu,
            s.description_en = row.description_en,
            s.category = row.category,
            s.severity = row.severity,
            s.possible_causes = row.possible_causes,
            s.diagnostic_steps = row.diagnostic_steps,
            s.keywords = row.keywords,
            s.created_at = datetime()
        ON MATCH SET
            s.name_hu = row.name_hu,
            s.name_en = row.name_en,
            s.description_hu = row.description_hu,
            s.description_en = row.description_en,
            s.category = row.category,
            s.severity = row.severity,
            s.possible_causes = row.possible_causes,
            s.diagnostic_steps = row.diagnostic_steps,
            s.keywords = row.keywords,
            s.updated_at = datetime()
        """

        data = []
        for symptom in symptoms:
            data.append({
                "id": symptom["id"],
                "name_hu": symptom.get("name_hu", ""),
                "name_en": symptom.get("name_en", ""),
                "description_hu": symptom.get("description_hu"),
                "description_en": symptom.get("description_en"),
                "category": symptom.get("category"),
                "severity": symptom.get("severity", "medium"),
                "possible_causes": symptom.get("possible_causes", []),
                "diagnostic_steps": symptom.get("diagnostic_steps", []),
                "keywords": symptom.get("keywords", []),
            })

        self._run_batch_query(query, data, desc="Symptom nodes")
        self.stats.symptom_nodes = len(data)
        return len(data)

    def seed_category_nodes(self, categories: List[Dict]) -> int:
        """Seed Category nodes from symptoms data."""
        logger.info("Seeding %d Category nodes...", len(categories))

        query = """
        UNWIND $batch AS row
        MERGE (c:Category {id: row.id})
        ON CREATE SET
            c.name_hu = row.name_hu,
            c.name_en = row.name_en,
            c.created_at = datetime()
        ON MATCH SET
            c.name_hu = row.name_hu,
            c.name_en = row.name_en,
            c.updated_at = datetime()
        """

        data = []
        for cat in categories:
            data.append({
                "id": cat["id"],
                "name_hu": cat.get("name_hu", ""),
                "name_en": cat.get("name_en", ""),
            })

        self._run_batch_query(query, data, desc="Category nodes")
        self.stats.category_nodes = len(data)
        return len(data)

    def seed_dtc_component_relationships(self, mappings: List[Dict]) -> int:
        """Create DTC -[:INDICATES_FAILURE_OF]-> Component relationships."""
        logger.info("Creating DTC -> Component relationships...")

        query = """
        UNWIND $batch AS row
        MATCH (d:DTC {code: row.dtc_code})
        MATCH (c:Component {id: row.component_id})
        MERGE (d)-[r:INDICATES_FAILURE_OF]->(c)
        ON CREATE SET
            r.confidence = row.confidence,
            r.failure_mode = row.failure_mode,
            r.created_at = datetime()
        ON MATCH SET
            r.confidence = row.confidence,
            r.failure_mode = row.failure_mode,
            r.updated_at = datetime()
        """

        # Flatten mappings (one mapping can have multiple components)
        data = []
        for mapping in mappings:
            dtc_pattern = mapping["dtc_pattern"]
            for component_id in mapping.get("components", []):
                data.append({
                    "dtc_code": dtc_pattern,
                    "component_id": component_id,
                    "confidence": mapping.get("confidence", 0.8),
                    "failure_mode": mapping.get("failure_mode", "malfunction"),
                })

        self._run_batch_query(query, data, desc="DTC->Component relationships")
        self.stats.dtc_component_rels = len(data)
        return len(data)

    def seed_component_repair_relationships(self, mappings: List[Dict]) -> int:
        """Create Component -[:REPAIRED_BY]-> Repair relationships."""
        logger.info("Creating Component -> Repair relationships...")

        query = """
        UNWIND $batch AS row
        MATCH (c:Component {id: row.component_id})
        MATCH (r:Repair {id: row.repair_id})
        MERGE (c)-[rel:REPAIRED_BY]->(r)
        ON CREATE SET
            rel.is_primary = row.is_primary,
            rel.created_at = datetime()
        ON MATCH SET
            rel.is_primary = row.is_primary,
            rel.updated_at = datetime()
        """

        # Flatten mappings
        data = []
        for mapping in mappings:
            component_id = mapping["component"]
            primary_repair = mapping.get("primary_repair")
            for repair_id in mapping.get("repairs", []):
                data.append({
                    "component_id": component_id,
                    "repair_id": repair_id,
                    "is_primary": repair_id == primary_repair,
                })

        self._run_batch_query(query, data, desc="Component->Repair relationships")
        self.stats.component_repair_rels = len(data)
        return len(data)

    def seed_dtc_symptom_relationships(self, symptoms: List[Dict]) -> int:
        """Create DTC -[:CAUSES]-> Symptom relationships based on related_dtc_codes in symptoms."""
        logger.info("Creating DTC -> Symptom relationships...")

        query = """
        UNWIND $batch AS row
        MATCH (d:DTC {code: row.dtc_code})
        MATCH (s:Symptom {id: row.symptom_id})
        MERGE (d)-[r:CAUSES]->(s)
        ON CREATE SET
            r.confidence = 0.7,
            r.created_at = datetime()
        """

        # Build relationships from symptoms' related_dtc_codes
        data = []
        for symptom in symptoms:
            symptom_id = symptom["id"]
            for dtc_code in symptom.get("related_dtc_codes", []):
                data.append({
                    "dtc_code": dtc_code,
                    "symptom_id": symptom_id,
                })

        self._run_batch_query(query, data, desc="DTC->Symptom relationships")
        self.stats.dtc_symptom_rels = len(data)
        return len(data)

    def seed_symptom_category_relationships(self, symptoms: List[Dict]) -> int:
        """Create Symptom -[:BELONGS_TO]-> Category relationships."""
        logger.info("Creating Symptom -> Category relationships...")

        query = """
        UNWIND $batch AS row
        MATCH (s:Symptom {id: row.symptom_id})
        MATCH (c:Category {id: row.category_id})
        MERGE (s)-[r:BELONGS_TO]->(c)
        ON CREATE SET
            r.created_at = datetime()
        """

        data = []
        for symptom in symptoms:
            if symptom.get("category"):
                data.append({
                    "symptom_id": symptom["id"],
                    "category_id": symptom["category"],
                })

        self._run_batch_query(query, data, desc="Symptom->Category relationships")
        self.stats.symptom_category_rels = len(data)
        return len(data)


def load_json_file(path: Path) -> Optional[Dict]:
    """Load JSON file and return contents."""
    if not path.exists():
        logger.warning("File not found: %s", path)
        return None

    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse JSON file %s: %s", path, e)
        return None


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Seed Neo4j Aura with AutoCognitix diagnostic graph data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/seed_neo4j_aura.py                    # Full seed
  python scripts/seed_neo4j_aura.py --dry-run         # Preview changes
  python scripts/seed_neo4j_aura.py --clear           # Clear and reseed
  python scripts/seed_neo4j_aura.py --nodes-only      # Seed only nodes
        """,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without modifying the database",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear all existing data before seeding",
    )
    parser.add_argument(
        "--nodes-only",
        action="store_true",
        help="Seed only nodes, skip relationships",
    )
    parser.add_argument(
        "--relationships-only",
        action="store_true",
        help="Seed only relationships, skip nodes",
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

    # Get connection details from environment
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD")

    if not neo4j_uri:
        logger.error("NEO4J_URI environment variable is required")
        logger.info("Example: export NEO4J_URI=neo4j+s://xxx.databases.neo4j.io")
        return 1

    if not neo4j_password:
        logger.error("NEO4J_PASSWORD environment variable is required")
        return 1

    # Load all data files
    logger.info("Loading data files...")

    dtc_data = load_json_file(DTC_CODES_PATH)
    if not dtc_data:
        logger.error("Failed to load DTC codes from %s", DTC_CODES_PATH)
        return 1
    dtc_codes = dtc_data.get("codes", [])
    logger.info("Loaded %d DTC codes", len(dtc_codes))

    components_data = load_json_file(COMPONENTS_PATH)
    if not components_data:
        logger.error("Failed to load components from %s", COMPONENTS_PATH)
        return 1
    components = components_data.get("components", [])
    logger.info("Loaded %d components", len(components))

    repairs_data = load_json_file(REPAIRS_PATH)
    if not repairs_data:
        logger.error("Failed to load repairs from %s", REPAIRS_PATH)
        return 1
    repairs = repairs_data.get("repairs", [])
    logger.info("Loaded %d repairs", len(repairs))

    dtc_component_mappings_data = load_json_file(DTC_COMPONENT_MAPPINGS_PATH)
    if not dtc_component_mappings_data:
        logger.error("Failed to load DTC-Component mappings from %s", DTC_COMPONENT_MAPPINGS_PATH)
        return 1
    dtc_component_mappings = dtc_component_mappings_data.get("mappings", [])
    logger.info("Loaded %d DTC-Component mappings", len(dtc_component_mappings))

    component_repair_mappings_data = load_json_file(COMPONENT_REPAIR_MAPPINGS_PATH)
    if not component_repair_mappings_data:
        logger.error(
            "Failed to load Component-Repair mappings from %s", COMPONENT_REPAIR_MAPPINGS_PATH
        )
        return 1
    component_repair_mappings = component_repair_mappings_data.get("mappings", [])
    logger.info("Loaded %d Component-Repair mappings", len(component_repair_mappings))

    # Symptoms are optional
    symptoms_data = load_json_file(SYMPTOMS_PATH)
    symptoms = []
    categories = []
    if symptoms_data:
        symptoms = symptoms_data.get("symptoms", [])
        categories = symptoms_data.get("categories", [])
        logger.info("Loaded %d symptoms and %d categories", len(symptoms), len(categories))
    else:
        logger.info("No symptoms file found, skipping symptom seeding")

    # Initialize seeder
    seeder = Neo4jSeeder(neo4j_uri, neo4j_user, neo4j_password, dry_run=args.dry_run)

    try:
        seeder.connect()

        if args.clear:
            seeder.clear_database()

        # Create constraints and indexes first
        seeder.create_constraints()
        seeder.create_indexes()

        # Seed nodes
        if not args.relationships_only:
            logger.info("\n=== Seeding Nodes ===")
            seeder.seed_dtc_nodes(dtc_codes)
            seeder.seed_component_nodes(components)
            seeder.seed_repair_nodes(repairs)

            if symptoms:
                seeder.seed_symptom_nodes(symptoms)

            if categories:
                seeder.seed_category_nodes(categories)

        # Seed relationships
        if not args.nodes_only:
            logger.info("\n=== Seeding Relationships ===")
            seeder.seed_dtc_component_relationships(dtc_component_mappings)
            seeder.seed_component_repair_relationships(component_repair_mappings)

            if symptoms:
                seeder.seed_dtc_symptom_relationships(symptoms)
                seeder.seed_symptom_category_relationships(symptoms)

        # Print summary
        print(seeder.stats.summary())

        if args.dry_run:
            logger.info("[DRY-RUN] No changes were made to the database")

        logger.info("Seeding completed successfully!")
        return 0

    except Exception as e:
        logger.error("Error during seeding: %s", e)
        raise
    finally:
        seeder.close()


if __name__ == "__main__":
    sys.exit(main())
