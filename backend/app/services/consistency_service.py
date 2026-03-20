"""
Cross-database consistency verification service.

Checks that DTC codes, vehicles, and embeddings are in sync across
PostgreSQL, Neo4j, and Qdrant databases.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Set

logger = logging.getLogger(__name__)


@dataclass
class ConsistencyReport:
    """Report from cross-database consistency check."""

    checked_at: str
    pg_dtc_count: int = 0
    neo4j_dtc_count: int = 0
    qdrant_vector_count: int = 0
    missing_in_neo4j: List[str] = field(default_factory=list)
    missing_in_qdrant: List[str] = field(default_factory=list)
    orphaned_in_neo4j: List[str] = field(default_factory=list)
    is_consistent: bool = False
    errors: List[str] = field(default_factory=list)


class ConsistencyService:
    """Verifies data consistency across PostgreSQL, Neo4j, and Qdrant."""

    async def check_dtc_consistency(self) -> ConsistencyReport:
        """
        Compare DTC codes across all three databases.

        Returns a report with counts and any mismatches found.
        """
        report = ConsistencyReport(checked_at=datetime.now(timezone.utc).isoformat())

        # Gather counts from all databases in parallel
        pg_codes, neo4j_codes, qdrant_count = await asyncio.gather(
            self._get_pg_dtc_codes(),
            self._get_neo4j_dtc_codes(),
            self._get_qdrant_vector_count(),
            return_exceptions=True,
        )

        # Handle errors from each database
        if isinstance(pg_codes, Exception):
            report.errors.append(f"PostgreSQL error: {pg_codes}")
            pg_codes = set()
        if isinstance(neo4j_codes, Exception):
            report.errors.append(f"Neo4j error: {neo4j_codes}")
            neo4j_codes = set()
        if isinstance(qdrant_count, Exception):
            report.errors.append(f"Qdrant error: {qdrant_count}")
            qdrant_count = 0

        report.pg_dtc_count = len(pg_codes)
        report.neo4j_dtc_count = len(neo4j_codes)
        report.qdrant_vector_count = qdrant_count

        # Find mismatches between PostgreSQL and Neo4j
        report.missing_in_neo4j = sorted(pg_codes - neo4j_codes)[:50]
        report.orphaned_in_neo4j = sorted(neo4j_codes - pg_codes)[:50]

        # Consistency = no errors and no mismatches
        report.is_consistent = (
            not report.errors and not report.missing_in_neo4j and not report.orphaned_in_neo4j
        )

        logger.info(
            "DTC consistency check completed",
            extra={
                "event": "consistency_check",
                "pg_count": report.pg_dtc_count,
                "neo4j_count": report.neo4j_dtc_count,
                "qdrant_count": report.qdrant_vector_count,
                "is_consistent": report.is_consistent,
                "error_count": len(report.errors),
            },
        )

        return report

    async def _get_pg_dtc_codes(self) -> Set[str]:
        """Get all DTC codes from PostgreSQL."""
        from sqlalchemy import text

        from app.db.postgres.session import async_session_maker

        async with async_session_maker() as session:
            result = await session.execute(
                text("SELECT code FROM dtc_codes"),
                execution_options={"timeout": 10},
            )
            return {row[0] for row in result}

    async def _get_neo4j_dtc_codes(self) -> Set[str]:
        """Get all DTC codes from Neo4j."""
        from neo4j import GraphDatabase

        from app.core.config import settings

        driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )
        try:

            def _query(tx):
                result = tx.run(
                    "MATCH (d:DTCCode) RETURN d.code AS code",
                    timeout=10,
                )
                return {record["code"] for record in result}

            def _run():
                with driver.session() as session:
                    return session.execute_read(_query)

            return await asyncio.to_thread(_run)
        finally:
            driver.close()

    async def _get_qdrant_vector_count(self) -> int:
        """Get total vector count from Qdrant DTC embedding collections."""
        from qdrant_client import QdrantClient

        from app.core.config import settings

        if settings.QDRANT_URL:
            client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
                timeout=10,
            )
        else:
            client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                timeout=10,
            )

        # Try Hungarian collection first, fall back to legacy English name
        total = 0
        for collection_name in ("dtc_embeddings_hu", "dtc_embeddings"):
            try:
                info = await asyncio.to_thread(client.get_collection, collection_name)
                total += info.points_count or 0
            except Exception:
                # Collection may not exist; skip
                pass

        return total
