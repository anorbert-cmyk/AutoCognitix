"""
AutoCognitix Data Export Utilities

This module provides comprehensive data export capabilities for the AutoCognitix platform.

Available export scripts:
- export_dtc_database.py - Export DTC codes to JSON, CSV, SQLite formats
- export_neo4j_graph.py - Export Neo4j graph to Cypher, JSON, GraphML, CSV formats
- export_qdrant_vectors.py - Export Qdrant vectors to NumPy, JSON formats
- export_full_backup.py - Create comprehensive system backups

Usage:
    # From command line
    python -m scripts.export.export_dtc_database --all
    python -m scripts.export.export_neo4j_graph --all
    python -m scripts.export.export_qdrant_vectors --all
    python -m scripts.export.export_full_backup

    # Or directly
    python scripts/export/export_dtc_database.py --all
"""

__version__ = "2.0.0"
__author__ = "AutoCognitix Team"
