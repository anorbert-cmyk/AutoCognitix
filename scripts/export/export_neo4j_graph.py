#!/usr/bin/env python3
"""
Neo4j Graph Export Utility for AutoCognitix

Exports Neo4j graph data to multiple formats:
- Cypher dump format (for neo4j-admin import or LOAD CSV)
- JSON format (structured nodes and relationships)
- GraphML format (for Gephi, yEd, Cytoscape)
- CSV format (separate files for nodes and relationships)

Supports exporting:
- DTC nodes
- Symptom nodes
- Component nodes
- Repair nodes
- Vehicle nodes
- Engine nodes
- All relationships between nodes

Usage:
    # Export all formats
    python scripts/export/export_neo4j_graph.py --all

    # Export specific format
    python scripts/export/export_neo4j_graph.py --cypher
    python scripts/export/export_neo4j_graph.py --json
    python scripts/export/export_neo4j_graph.py --graphml
    python scripts/export/export_neo4j_graph.py --csv

    # Export specific node types
    python scripts/export/export_neo4j_graph.py --all --nodes DTC,Symptom

    # Export with relationship filtering
    python scripts/export/export_neo4j_graph.py --all --relationships CAUSES,INDICATES_FAILURE_OF
"""

import argparse
import csv
import json
import logging
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from xml.dom import minidom

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

# Neo4j node labels used in AutoCognitix
NODE_LABELS = [
    "DTCNode",
    "SymptomNode",
    "ComponentNode",
    "RepairNode",
    "PartNode",
    "TestPointNode",
    "VehicleNode",
    "EngineNode",
    "PlatformNode",
]

# Relationship types used in AutoCognitix
RELATIONSHIP_TYPES = [
    "CAUSES",
    "INDICATES_FAILURE_OF",
    "REPAIRED_BY",
    "USES_PART",
    "REQUIRES_CHECK",
    "LEADS_TO",
    "HAS_COMMON_ISSUE",
    "USES_COMPONENT",
    "USES_ENGINE",
    "COMMON_REPAIR",
    "SHARES_PLATFORM",
    "RELATED_TO",
    "CONTAINS",
]


class Neo4jExporter:
    """Handles exporting Neo4j graph data to various formats."""

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        node_filter: Optional[List[str]] = None,
        relationship_filter: Optional[List[str]] = None,
    ):
        """
        Initialize the Neo4j exporter.

        Args:
            output_dir: Output directory for exports
            node_filter: Filter by node labels
            relationship_filter: Filter by relationship types
        """
        self.output_dir = output_dir or EXPORT_DIR
        self.node_filter = node_filter
        self.relationship_filter = relationship_filter

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data containers
        self.nodes: Dict[str, List[Dict[str, Any]]] = {}
        self.relationships: List[Dict[str, Any]] = []

        # Statistics
        self.stats = {
            "total_nodes": 0,
            "total_relationships": 0,
            "nodes_by_label": {},
            "relationships_by_type": {},
        }

        # Track node element IDs for relationship mapping
        self.node_id_map: Dict[str, str] = {}  # element_id -> export_id

    def connect(self):
        """
        Connect to Neo4j database.

        Returns:
            Neo4j driver instance
        """
        try:
            from neo4j import GraphDatabase
            from backend.app.core.config import settings

            driver = GraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
            )

            # Test connection
            with driver.session() as session:
                session.run("RETURN 1")

            logger.info(f"Connected to Neo4j at {settings.NEO4J_URI}")
            return driver

        except ImportError:
            logger.error("neo4j package not installed. Run: pip install neo4j")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def load_data(self) -> None:
        """Load all nodes and relationships from Neo4j."""
        logger.info("Loading data from Neo4j...")

        driver = self.connect()

        with driver.session() as session:
            # Get available labels
            labels_result = session.run("CALL db.labels()")
            available_labels = [record["label"] for record in labels_result]
            logger.info(f"Available labels: {available_labels}")

            # Determine which labels to export
            labels_to_export = self.node_filter or available_labels

            # Export nodes by label
            node_counter = 0
            for label in labels_to_export:
                if label not in available_labels:
                    logger.warning(f"Label '{label}' not found in database, skipping")
                    continue

                # Security: validate label name
                if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', label):
                    logger.warning(f"Skipping invalid label name: {label}")
                    continue

                logger.info(f"  Loading nodes: {label}")
                result = session.run(
                    f"MATCH (n:{label}) RETURN n, elementId(n) as element_id"
                )

                nodes = []
                for record in result:
                    node = record["n"]
                    element_id = record["element_id"]

                    # Create export node
                    node_data = dict(node)
                    node_data["_element_id"] = element_id
                    node_data["_label"] = label

                    # Map element_id to export_id
                    export_id = f"n{node_counter}"
                    self.node_id_map[element_id] = export_id
                    node_data["_export_id"] = export_id
                    node_counter += 1

                    nodes.append(node_data)

                self.nodes[label] = nodes
                self.stats["nodes_by_label"][label] = len(nodes)
                self.stats["total_nodes"] += len(nodes)

                logger.info(f"    Loaded {len(nodes)} nodes")

            # Export relationships
            logger.info("  Loading relationships...")

            rel_query = """
                MATCH (a)-[r]->(b)
                RETURN
                    labels(a)[0] as start_label,
                    elementId(a) as start_id,
                    type(r) as rel_type,
                    properties(r) as rel_props,
                    labels(b)[0] as end_label,
                    elementId(b) as end_id,
                    elementId(r) as rel_id
            """
            result = session.run(rel_query)

            for record in result:
                rel_type = record["rel_type"]

                # Apply relationship filter
                if self.relationship_filter and rel_type not in self.relationship_filter:
                    continue

                # Skip relationships to/from filtered nodes
                start_id = record["start_id"]
                end_id = record["end_id"]

                if start_id not in self.node_id_map or end_id not in self.node_id_map:
                    continue

                rel_data = {
                    "start_label": record["start_label"],
                    "start_id": start_id,
                    "start_export_id": self.node_id_map.get(start_id),
                    "type": rel_type,
                    "properties": dict(record["rel_props"]) if record["rel_props"] else {},
                    "end_label": record["end_label"],
                    "end_id": end_id,
                    "end_export_id": self.node_id_map.get(end_id),
                    "rel_id": record["rel_id"],
                }

                self.relationships.append(rel_data)

                # Update stats
                if rel_type not in self.stats["relationships_by_type"]:
                    self.stats["relationships_by_type"][rel_type] = 0
                self.stats["relationships_by_type"][rel_type] += 1

            self.stats["total_relationships"] = len(self.relationships)
            logger.info(f"    Loaded {len(self.relationships)} relationships")

        driver.close()
        logger.info(f"Total loaded: {self.stats['total_nodes']} nodes, {self.stats['total_relationships']} relationships")

    def export_to_cypher(self) -> Path:
        """
        Export graph to Cypher dump format.

        Creates a .cypher file that can be executed to recreate the graph.

        Returns:
            Path to exported file
        """
        logger.info("Exporting to Cypher format...")

        output_file = self.output_dir / "neo4j_dump.cypher"

        with open(output_file, 'w', encoding='utf-8') as f:
            # Header
            f.write("// AutoCognitix Neo4j Graph Export\n")
            f.write(f"// Exported at: {datetime.now().isoformat()}\n")
            f.write(f"// Total nodes: {self.stats['total_nodes']}\n")
            f.write(f"// Total relationships: {self.stats['total_relationships']}\n")
            f.write("\n")

            # Clear existing data (optional, commented by default)
            f.write("// Uncomment the following lines to clear existing data:\n")
            f.write("// MATCH (n) DETACH DELETE n;\n")
            f.write("\n")

            # Create constraints and indexes
            f.write("// Create constraints and indexes\n")
            for label in self.nodes.keys():
                # Determine the unique property
                if label == "DTCNode":
                    f.write(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.code IS UNIQUE;\n")
                elif label == "EngineNode":
                    f.write(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.code IS UNIQUE;\n")
                elif label == "PlatformNode":
                    f.write(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.code IS UNIQUE;\n")
                else:
                    f.write(f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.name);\n")
            f.write("\n")

            # Create nodes
            f.write("// Create nodes\n")
            for label, nodes in self.nodes.items():
                f.write(f"\n// {label} nodes ({len(nodes)})\n")

                for node in tqdm(nodes, desc=f"Writing {label} nodes", disable=not HAS_TQDM):
                    props = {k: v for k, v in node.items() if not k.startswith("_")}
                    props_str = self._format_cypher_properties(props)

                    if props_str:
                        f.write(f"CREATE (:{label} {props_str});\n")
                    else:
                        f.write(f"CREATE (:{label});\n")

            f.write("\n")

            # Create relationships
            f.write("// Create relationships\n")
            for rel_type in set(r["type"] for r in self.relationships):
                rels_of_type = [r for r in self.relationships if r["type"] == rel_type]
                f.write(f"\n// {rel_type} relationships ({len(rels_of_type)})\n")

                for rel in tqdm(rels_of_type, desc=f"Writing {rel_type} relationships", disable=not HAS_TQDM):
                    start_label = rel["start_label"]
                    end_label = rel["end_label"]
                    props = rel["properties"]

                    # Build MATCH clause based on available unique properties
                    start_match = self._build_match_clause("a", start_label, rel["start_id"])
                    end_match = self._build_match_clause("b", end_label, rel["end_id"])

                    if start_match and end_match:
                        props_str = self._format_cypher_properties(props)
                        if props_str:
                            f.write(f"MATCH {start_match} MATCH {end_match} MERGE (a)-[:{rel_type} {props_str}]->(b);\n")
                        else:
                            f.write(f"MATCH {start_match} MATCH {end_match} MERGE (a)-[:{rel_type}]->(b);\n")

        file_size = output_file.stat().st_size / 1024
        logger.info(f"Cypher export complete: {output_file} ({file_size:.1f} KB)")

        return output_file

    def _format_cypher_properties(self, props: Dict[str, Any]) -> str:
        """Format properties as Cypher map."""
        if not props:
            return ""

        formatted_props = []
        for key, value in props.items():
            if value is None:
                continue

            # Escape key if needed
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', key):
                key = f"`{key}`"

            # Format value
            if isinstance(value, str):
                escaped = value.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')
                formatted_props.append(f"{key}: '{escaped}'")
            elif isinstance(value, bool):
                formatted_props.append(f"{key}: {str(value).lower()}")
            elif isinstance(value, (int, float)):
                formatted_props.append(f"{key}: {value}")
            elif isinstance(value, list):
                list_str = json.dumps(value)
                formatted_props.append(f"{key}: {list_str}")
            else:
                # Convert to string
                str_value = str(value).replace("'", "\\'")
                formatted_props.append(f"{key}: '{str_value}'")

        return "{" + ", ".join(formatted_props) + "}" if formatted_props else ""

    def _build_match_clause(self, alias: str, label: str, element_id: str) -> Optional[str]:
        """Build MATCH clause for a node."""
        # Find the node by element_id
        nodes = self.nodes.get(label, [])
        node = None
        for n in nodes:
            if n.get("_element_id") == element_id:
                node = n
                break

        if not node:
            return None

        # Use appropriate unique property for matching
        if label == "DTCNode" and node.get("code"):
            return f"({alias}:{label} {{code: '{node['code']}'}})"
        elif label == "EngineNode" and node.get("code"):
            return f"({alias}:{label} {{code: '{node['code']}'}})"
        elif label == "PlatformNode" and node.get("code"):
            return f"({alias}:{label} {{code: '{node['code']}'}})"
        elif node.get("name"):
            name = node["name"].replace("'", "\\'")
            return f"({alias}:{label} {{name: '{name}'}})"
        elif node.get("uid"):
            return f"({alias}:{label} {{uid: '{node['uid']}'}})"

        return None

    def export_to_json(self) -> Path:
        """
        Export graph to JSON format.

        Returns:
            Path to exported file
        """
        logger.info("Exporting to JSON format...")

        export_data = {
            "export_info": {
                "export_time": datetime.now().isoformat(),
                "source": "AutoCognitix",
                "version": "2.0",
                "statistics": self.stats,
            },
            "nodes": {},
            "relationships": [],
        }

        # Process nodes (remove internal fields)
        for label, nodes in self.nodes.items():
            cleaned_nodes = []
            for node in tqdm(nodes, desc=f"Processing {label} nodes", disable=not HAS_TQDM):
                cleaned = {k: v for k, v in node.items() if not k.startswith("_")}
                cleaned["_id"] = node.get("_export_id")
                cleaned["_label"] = label
                cleaned_nodes.append(cleaned)
            export_data["nodes"][label] = cleaned_nodes

        # Process relationships
        for rel in tqdm(self.relationships, desc="Processing relationships", disable=not HAS_TQDM):
            export_data["relationships"].append({
                "type": rel["type"],
                "start_id": rel["start_export_id"],
                "start_label": rel["start_label"],
                "end_id": rel["end_export_id"],
                "end_label": rel["end_label"],
                "properties": rel["properties"],
            })

        output_file = self.output_dir / "neo4j_graph.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)

        file_size = output_file.stat().st_size / 1024
        logger.info(f"JSON export complete: {output_file} ({file_size:.1f} KB)")

        return output_file

    def export_to_graphml(self) -> Path:
        """
        Export graph to GraphML format.

        GraphML can be imported into:
        - Gephi
        - yEd
        - Cytoscape
        - Neo4j (via APOC)

        Returns:
            Path to exported file
        """
        logger.info("Exporting to GraphML format...")

        # Create GraphML root element
        graphml = ET.Element("graphml")
        graphml.set("xmlns", "http://graphml.graphdrawing.org/xmlns")
        graphml.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
        graphml.set("xsi:schemaLocation",
                    "http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd")

        # Collect all unique property keys
        node_keys: Set[str] = set()
        rel_keys: Set[str] = set()

        for label, nodes in self.nodes.items():
            for node in nodes:
                for key in node.keys():
                    if not key.startswith("_"):
                        node_keys.add(key)

        for rel in self.relationships:
            for key in rel.get("properties", {}).keys():
                rel_keys.add(key)

        # Define key elements for node properties
        for key_name in sorted(node_keys):
            key_elem = ET.SubElement(graphml, "key")
            key_elem.set("id", f"n_{key_name}")
            key_elem.set("for", "node")
            key_elem.set("attr.name", key_name)
            key_elem.set("attr.type", "string")

        # Add label key for nodes
        label_key = ET.SubElement(graphml, "key")
        label_key.set("id", "n_label")
        label_key.set("for", "node")
        label_key.set("attr.name", "label")
        label_key.set("attr.type", "string")

        # Define key elements for edge properties
        for key_name in sorted(rel_keys):
            key_elem = ET.SubElement(graphml, "key")
            key_elem.set("id", f"e_{key_name}")
            key_elem.set("for", "edge")
            key_elem.set("attr.name", key_name)
            key_elem.set("attr.type", "string")

        # Add relationship type key
        rel_type_key = ET.SubElement(graphml, "key")
        rel_type_key.set("id", "e_type")
        rel_type_key.set("for", "edge")
        rel_type_key.set("attr.name", "type")
        rel_type_key.set("attr.type", "string")

        # Create graph element
        graph = ET.SubElement(graphml, "graph")
        graph.set("id", "G")
        graph.set("edgedefault", "directed")

        # Add nodes
        for label, nodes in self.nodes.items():
            for node in tqdm(nodes, desc=f"Adding {label} nodes to GraphML", disable=not HAS_TQDM):
                node_id = node.get("_export_id", "")
                node_elem = ET.SubElement(graph, "node")
                node_elem.set("id", node_id)

                # Add label
                label_data = ET.SubElement(node_elem, "data")
                label_data.set("key", "n_label")
                label_data.text = label

                # Add properties
                for key, value in node.items():
                    if key.startswith("_"):
                        continue
                    data_elem = ET.SubElement(node_elem, "data")
                    data_elem.set("key", f"n_{key}")
                    if isinstance(value, (list, dict)):
                        data_elem.text = json.dumps(value, ensure_ascii=False)
                    else:
                        data_elem.text = str(value) if value is not None else ""

        # Add edges (relationships)
        edge_counter = 0
        for rel in tqdm(self.relationships, desc="Adding relationships to GraphML", disable=not HAS_TQDM):
            source_id = rel.get("start_export_id")
            target_id = rel.get("end_export_id")

            if not source_id or not target_id:
                continue

            edge_id = f"e{edge_counter}"
            edge_counter += 1

            edge_elem = ET.SubElement(graph, "edge")
            edge_elem.set("id", edge_id)
            edge_elem.set("source", source_id)
            edge_elem.set("target", target_id)

            # Add relationship type
            type_data = ET.SubElement(edge_elem, "data")
            type_data.set("key", "e_type")
            type_data.text = rel["type"]

            # Add properties
            for key, value in rel.get("properties", {}).items():
                data_elem = ET.SubElement(edge_elem, "data")
                data_elem.set("key", f"e_{key}")
                if isinstance(value, (list, dict)):
                    data_elem.text = json.dumps(value, ensure_ascii=False)
                else:
                    data_elem.text = str(value) if value is not None else ""

        # Write to file with pretty printing
        output_file = self.output_dir / "neo4j_graph.graphml"

        rough_string = ET.tostring(graphml, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")

        # Remove extra blank lines
        lines = [line for line in pretty_xml.split('\n') if line.strip()]
        pretty_xml = '\n'.join(lines)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(pretty_xml)

        file_size = output_file.stat().st_size / 1024
        logger.info(f"GraphML export complete: {output_file} ({file_size:.1f} KB)")

        return output_file

    def export_to_csv(self) -> List[Path]:
        """
        Export graph to CSV format.

        Creates separate CSV files for each node type and relationships.

        Returns:
            List of paths to exported files
        """
        logger.info("Exporting to CSV format...")

        exported_files = []

        # Create CSV subdirectory
        csv_dir = self.output_dir / "csv"
        csv_dir.mkdir(exist_ok=True)

        # Export nodes by label
        for label, nodes in self.nodes.items():
            if not nodes:
                continue

            # Get all unique keys
            all_keys = set()
            for node in nodes:
                for key in node.keys():
                    if not key.startswith("_") or key == "_export_id":
                        all_keys.add(key)
            all_keys = sorted(all_keys)

            output_file = csv_dir / f"nodes_{label}.csv"
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=all_keys)
                writer.writeheader()

                for node in tqdm(nodes, desc=f"Writing {label} CSV", disable=not HAS_TQDM):
                    row = {}
                    for key in all_keys:
                        value = node.get(key, "")
                        if isinstance(value, (list, dict)):
                            value = json.dumps(value, ensure_ascii=False)
                        row[key] = value
                    writer.writerow(row)

            logger.info(f"  Exported {label}: {len(nodes)} nodes")
            exported_files.append(output_file)

        # Export relationships
        if self.relationships:
            output_file = csv_dir / "relationships.csv"
            fieldnames = ["start_id", "start_label", "type", "end_id", "end_label", "properties"]

            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for rel in tqdm(self.relationships, desc="Writing relationships CSV", disable=not HAS_TQDM):
                    writer.writerow({
                        "start_id": rel["start_export_id"],
                        "start_label": rel["start_label"],
                        "type": rel["type"],
                        "end_id": rel["end_export_id"],
                        "end_label": rel["end_label"],
                        "properties": json.dumps(rel["properties"], ensure_ascii=False),
                    })

            logger.info(f"  Exported relationships: {len(self.relationships)}")
            exported_files.append(output_file)

        logger.info(f"CSV export complete: {len(exported_files)} files in {csv_dir}")

        return exported_files

    def print_statistics(self) -> None:
        """Print export statistics."""
        logger.info("=" * 50)
        logger.info("EXPORT STATISTICS")
        logger.info("=" * 50)
        logger.info(f"Total nodes: {self.stats['total_nodes']}")
        logger.info(f"Total relationships: {self.stats['total_relationships']}")
        logger.info("")
        logger.info("Nodes by label:")
        for label, count in sorted(self.stats["nodes_by_label"].items()):
            logger.info(f"  {label}: {count}")
        logger.info("")
        logger.info("Relationships by type:")
        for rel_type, count in sorted(self.stats["relationships_by_type"].items()):
            logger.info(f"  {rel_type}: {count}")
        logger.info("=" * 50)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Export Neo4j graph to multiple formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python export_neo4j_graph.py --all
    python export_neo4j_graph.py --cypher --json
    python export_neo4j_graph.py --all --nodes DTCNode,SymptomNode
    python export_neo4j_graph.py --all --relationships CAUSES,INDICATES_FAILURE_OF
        """
    )

    # Format options
    format_group = parser.add_argument_group("Export Formats")
    format_group.add_argument(
        "--cypher",
        action="store_true",
        help="Export to Cypher dump format (neo4j_dump.cypher)",
    )
    format_group.add_argument(
        "--json",
        action="store_true",
        help="Export to JSON format (neo4j_graph.json)",
    )
    format_group.add_argument(
        "--graphml",
        action="store_true",
        help="Export to GraphML format (neo4j_graph.graphml)",
    )
    format_group.add_argument(
        "--csv",
        action="store_true",
        help="Export to CSV format (csv/*.csv)",
    )
    format_group.add_argument(
        "--all",
        action="store_true",
        help="Export to all formats",
    )

    # Filter options
    filter_group = parser.add_argument_group("Filtering Options")
    filter_group.add_argument(
        "--nodes",
        type=str,
        help=f"Filter by node labels (comma-separated). Available: {', '.join(NODE_LABELS)}",
    )
    filter_group.add_argument(
        "--relationships",
        type=str,
        help=f"Filter by relationship types (comma-separated). Available: {', '.join(RELATIONSHIP_TYPES)}",
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
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse filters
    node_filter = None
    if args.nodes:
        node_filter = [n.strip() for n in args.nodes.split(",")]

    relationship_filter = None
    if args.relationships:
        relationship_filter = [r.strip() for r in args.relationships.split(",")]

    # Setup output directory
    output_dir = None
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Create exporter
    exporter = Neo4jExporter(
        output_dir=output_dir,
        node_filter=node_filter,
        relationship_filter=relationship_filter,
    )

    # Load data
    try:
        exporter.load_data()
    except Exception as e:
        logger.error(f"Failed to load data from Neo4j: {e}")
        sys.exit(1)

    if exporter.stats["total_nodes"] == 0:
        logger.warning("No nodes found in Neo4j database!")
        sys.exit(1)

    # Default to --all if no specific format is selected
    if not any([args.cypher, args.json, args.graphml, args.csv, args.all]):
        args.all = True

    # Export to requested formats
    exported_files = []

    try:
        if args.cypher or args.all:
            exported_files.append(exporter.export_to_cypher())

        if args.json or args.all:
            exported_files.append(exporter.export_to_json())

        if args.graphml or args.all:
            exported_files.append(exporter.export_to_graphml())

        if args.csv or args.all:
            exported_files.extend(exporter.export_to_csv())

        # Print statistics
        exporter.print_statistics()

        # Summary
        logger.info("")
        logger.info("Exported files:")
        for f in exported_files:
            if isinstance(f, Path):
                logger.info(f"  {f}")
            else:
                logger.info(f"  {f}")

        logger.info("")
        logger.info("Export completed successfully!")

    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise


if __name__ == "__main__":
    main()
