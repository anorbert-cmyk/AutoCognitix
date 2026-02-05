#!/usr/bin/env python3
"""
Health Check Script for AutoCognitix

This script checks the health of all system components:
- PostgreSQL database connection and basic queries
- Neo4j graph database connection and state
- Qdrant vector database collection status
- Redis cache connection
- Backend API health endpoints

Usage:
    python scripts/health_check.py           # Check all services
    python scripts/health_check.py --postgres # Check only PostgreSQL
    python scripts/health_check.py --neo4j    # Check only Neo4j
    python scripts/health_check.py --qdrant   # Check only Qdrant
    python scripts/health_check.py --redis    # Check only Redis
    python scripts/health_check.py --api      # Check only API
    python scripts/health_check.py --json     # Output as JSON
"""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import settings


@dataclass
class ServiceHealth:
    """Health status for a single service."""
    name: str
    status: str  # "healthy", "degraded", "unhealthy", "unknown"
    latency_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    checked_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class HealthReport:
    """Complete health report for all services."""
    overall_status: str = "unknown"
    services: List[ServiceHealth] = field(default_factory=list)
    checked_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    environment: str = settings.ENVIRONMENT

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_status": self.overall_status,
            "services": [asdict(s) for s in self.services],
            "checked_at": self.checked_at,
            "environment": self.environment,
        }


class HealthChecker:
    """Health checker for all AutoCognitix services."""

    def __init__(self):
        self.report = HealthReport()

    async def check_postgres(self) -> ServiceHealth:
        """Check PostgreSQL database health."""
        from sqlalchemy import create_engine, text
        from sqlalchemy.exc import SQLAlchemyError

        start_time = time.time()
        try:
            # Convert async URL to sync
            db_url = settings.DATABASE_URL
            if db_url.startswith("postgresql+asyncpg://"):
                db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")

            engine = create_engine(db_url, pool_pre_ping=True)

            with engine.connect() as conn:
                # Basic connectivity test
                result = conn.execute(text("SELECT 1"))
                result.fetchone()

                # Get database version
                version_result = conn.execute(text("SELECT version()"))
                db_version = version_result.fetchone()[0]

                # Get table counts
                table_counts = {}
                for table in ["dtc_codes", "vehicle_makes", "vehicle_models", "users"]:
                    try:
                        count_result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                        table_counts[table] = count_result.fetchone()[0]
                    except SQLAlchemyError:
                        table_counts[table] = "table not found"

                # Get database size
                size_result = conn.execute(text(
                    "SELECT pg_size_pretty(pg_database_size(current_database()))"
                ))
                db_size = size_result.fetchone()[0]

            latency = (time.time() - start_time) * 1000

            return ServiceHealth(
                name="PostgreSQL",
                status="healthy",
                latency_ms=round(latency, 2),
                details={
                    "version": db_version.split(",")[0] if db_version else "unknown",
                    "database_size": db_size,
                    "table_counts": table_counts,
                    "host": settings.DATABASE_URL.split("@")[1].split("/")[0] if "@" in settings.DATABASE_URL else "localhost",
                }
            )

        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return ServiceHealth(
                name="PostgreSQL",
                status="unhealthy",
                latency_ms=round(latency, 2),
                error=str(e),
            )

    async def check_neo4j(self) -> ServiceHealth:
        """Check Neo4j database health."""
        from neo4j import GraphDatabase
        from neo4j.exceptions import ServiceUnavailable, AuthError

        start_time = time.time()
        try:
            driver = GraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
            )

            with driver.session() as session:
                # Basic connectivity test
                result = session.run("RETURN 1 AS num")
                result.single()

                # Get database info
                db_info = session.run("CALL dbms.components() YIELD name, versions, edition")
                component = db_info.single()

                # Get node counts by label
                node_counts = {}
                for label in ["DTCCode", "Symptom", "Component", "Repair"]:
                    try:
                        count_result = session.run(f"MATCH (n:{label}) RETURN COUNT(n) AS count")
                        node_counts[label] = count_result.single()["count"]
                    except Exception:
                        node_counts[label] = 0

                # Get relationship counts
                rel_result = session.run("MATCH ()-[r]->() RETURN COUNT(r) AS count")
                rel_count = rel_result.single()["count"]

            driver.close()
            latency = (time.time() - start_time) * 1000

            return ServiceHealth(
                name="Neo4j",
                status="healthy",
                latency_ms=round(latency, 2),
                details={
                    "version": component["versions"][0] if component else "unknown",
                    "edition": component["edition"] if component else "unknown",
                    "node_counts": node_counts,
                    "relationship_count": rel_count,
                    "uri": settings.NEO4J_URI,
                }
            )

        except AuthError as e:
            latency = (time.time() - start_time) * 1000
            return ServiceHealth(
                name="Neo4j",
                status="unhealthy",
                latency_ms=round(latency, 2),
                error=f"Authentication failed: {str(e)}",
            )
        except ServiceUnavailable as e:
            latency = (time.time() - start_time) * 1000
            return ServiceHealth(
                name="Neo4j",
                status="unhealthy",
                latency_ms=round(latency, 2),
                error=f"Service unavailable: {str(e)}",
            )
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return ServiceHealth(
                name="Neo4j",
                status="unhealthy",
                latency_ms=round(latency, 2),
                error=str(e),
            )

    async def check_qdrant(self) -> ServiceHealth:
        """Check Qdrant vector database health."""
        from qdrant_client import QdrantClient
        from qdrant_client.http.exceptions import UnexpectedResponse

        start_time = time.time()
        try:
            # Connect to Qdrant (skip version check for compatibility)
            if settings.QDRANT_URL:
                client = QdrantClient(
                    url=settings.QDRANT_URL,
                    api_key=settings.QDRANT_API_KEY,
                )
                qdrant_location = settings.QDRANT_URL
            else:
                client = QdrantClient(
                    host=settings.QDRANT_HOST,
                    port=settings.QDRANT_PORT,
                    check_compatibility=False,
                )
                qdrant_location = f"{settings.QDRANT_HOST}:{settings.QDRANT_PORT}"

            # Get collections info
            collections = client.get_collections()
            collection_stats = {}

            for collection in collections.collections:
                try:
                    info = client.get_collection(collection.name)
                    # Handle both old and new API versions
                    vectors_count = getattr(info, 'vectors_count', None)
                    if vectors_count is None:
                        vectors_count = getattr(info, 'points_count', 0)
                    points_count = getattr(info, 'points_count', vectors_count)
                    collection_stats[collection.name] = {
                        "vectors_count": vectors_count,
                        "points_count": points_count,
                        "status": str(info.status) if hasattr(info, 'status') else "unknown",
                    }
                except Exception as e:
                    collection_stats[collection.name] = {"error": str(e)}

            latency = (time.time() - start_time) * 1000

            return ServiceHealth(
                name="Qdrant",
                status="healthy",
                latency_ms=round(latency, 2),
                details={
                    "location": qdrant_location,
                    "collections_count": len(collections.collections),
                    "collections": collection_stats,
                }
            )

        except UnexpectedResponse as e:
            latency = (time.time() - start_time) * 1000
            return ServiceHealth(
                name="Qdrant",
                status="unhealthy",
                latency_ms=round(latency, 2),
                error=f"Unexpected response: {str(e)}",
            )
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return ServiceHealth(
                name="Qdrant",
                status="unhealthy",
                latency_ms=round(latency, 2),
                error=str(e),
            )

    async def check_redis(self) -> ServiceHealth:
        """Check Redis cache health."""
        start_time = time.time()
        try:
            import redis
        except ImportError:
            latency = (time.time() - start_time) * 1000
            return ServiceHealth(
                name="Redis",
                status="unhealthy",
                latency_ms=round(latency, 2),
                error="redis module not installed. Run: pip install redis",
            )

        try:
            client = redis.from_url(settings.REDIS_URL)

            # Basic connectivity test
            pong = client.ping()

            # Get Redis info
            info = client.info()

            # Get memory usage
            memory = client.info("memory")

            latency = (time.time() - start_time) * 1000

            return ServiceHealth(
                name="Redis",
                status="healthy" if pong else "degraded",
                latency_ms=round(latency, 2),
                details={
                    "version": info.get("redis_version", "unknown"),
                    "connected_clients": info.get("connected_clients", 0),
                    "used_memory_human": memory.get("used_memory_human", "unknown"),
                    "used_memory_peak_human": memory.get("used_memory_peak_human", "unknown"),
                    "total_keys": sum(info.get(f"db{i}", {}).get("keys", 0) for i in range(16)),
                    "url": settings.REDIS_URL.split("@")[-1] if "@" in settings.REDIS_URL else settings.REDIS_URL,
                }
            )

        except redis.ConnectionError as e:
            latency = (time.time() - start_time) * 1000
            return ServiceHealth(
                name="Redis",
                status="unhealthy",
                latency_ms=round(latency, 2),
                error=f"Connection failed: {str(e)}",
            )
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return ServiceHealth(
                name="Redis",
                status="unhealthy",
                latency_ms=round(latency, 2),
                error=str(e),
            )

    async def check_api(self, base_url: str = "http://localhost:8000") -> ServiceHealth:
        """Check Backend API health."""
        start_time = time.time()
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Check basic health endpoint
                health_response = await client.get(f"{base_url}/health")
                health_data = health_response.json()

                # Check detailed health if available
                detailed_data = None
                try:
                    detailed_response = await client.get(f"{base_url}/health/detailed")
                    if detailed_response.status_code == 200:
                        detailed_data = detailed_response.json()
                except Exception:
                    pass

                # Check OpenAPI docs availability
                docs_available = False
                try:
                    docs_response = await client.get(f"{base_url}/api/v1/docs")
                    docs_available = docs_response.status_code == 200
                except Exception:
                    pass

                latency = (time.time() - start_time) * 1000

                status = "healthy" if health_response.status_code == 200 else "degraded"

                return ServiceHealth(
                    name="Backend API",
                    status=status,
                    latency_ms=round(latency, 2),
                    details={
                        "base_url": base_url,
                        "health_status": health_data.get("status", "unknown"),
                        "version": health_data.get("version", "unknown"),
                        "service": health_data.get("service", "unknown"),
                        "docs_available": docs_available,
                        "detailed_health": detailed_data,
                    }
                )

        except httpx.ConnectError:
            latency = (time.time() - start_time) * 1000
            return ServiceHealth(
                name="Backend API",
                status="unhealthy",
                latency_ms=round(latency, 2),
                error=f"Cannot connect to API at {base_url}",
            )
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return ServiceHealth(
                name="Backend API",
                status="unhealthy",
                latency_ms=round(latency, 2),
                error=str(e),
            )

    async def run_all_checks(
        self,
        check_postgres: bool = True,
        check_neo4j: bool = True,
        check_qdrant: bool = True,
        check_redis: bool = True,
        check_api: bool = True,
        api_url: str = "http://localhost:8000",
    ) -> HealthReport:
        """Run all health checks."""
        tasks = []

        if check_postgres:
            tasks.append(self.check_postgres())
        if check_neo4j:
            tasks.append(self.check_neo4j())
        if check_qdrant:
            tasks.append(self.check_qdrant())
        if check_redis:
            tasks.append(self.check_redis())
        if check_api:
            tasks.append(self.check_api(api_url))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                self.report.services.append(ServiceHealth(
                    name="Unknown",
                    status="unhealthy",
                    error=str(result),
                ))
            else:
                self.report.services.append(result)

        # Determine overall status
        statuses = [s.status for s in self.report.services]
        if all(s == "healthy" for s in statuses):
            self.report.overall_status = "healthy"
        elif any(s == "unhealthy" for s in statuses):
            self.report.overall_status = "unhealthy"
        elif any(s == "degraded" for s in statuses):
            self.report.overall_status = "degraded"
        else:
            self.report.overall_status = "unknown"

        return self.report


def print_report(report: HealthReport, as_json: bool = False):
    """Print health report to console."""
    if as_json:
        print(json.dumps(report.to_dict(), indent=2, default=str))
        return

    # Color codes
    COLORS = {
        "healthy": "\033[92m",  # Green
        "degraded": "\033[93m",  # Yellow
        "unhealthy": "\033[91m",  # Red
        "unknown": "\033[90m",  # Gray
        "reset": "\033[0m",
        "bold": "\033[1m",
    }

    def colorize(status: str, text: str) -> str:
        return f"{COLORS.get(status, '')}{text}{COLORS['reset']}"

    print("\n" + "=" * 60)
    print(f"{COLORS['bold']}AutoCognitix Health Check Report{COLORS['reset']}")
    print("=" * 60)
    print(f"Environment: {report.environment}")
    print(f"Checked at:  {report.checked_at}")
    print("-" * 60)

    # Overall status
    overall_color = colorize(report.overall_status, report.overall_status.upper())
    print(f"\n{COLORS['bold']}Overall Status: {overall_color}")
    print("-" * 60)

    # Service details
    for service in report.services:
        status_display = colorize(service.status, service.status.upper())
        print(f"\n{COLORS['bold']}{service.name}{COLORS['reset']}")
        print(f"  Status:  {status_display}")
        print(f"  Latency: {service.latency_ms}ms")

        if service.error:
            print(f"  {COLORS['unhealthy']}Error: {service.error}{COLORS['reset']}")

        if service.details:
            print("  Details:")
            for key, value in service.details.items():
                if isinstance(value, dict):
                    print(f"    {key}:")
                    for k, v in value.items():
                        print(f"      {k}: {v}")
                else:
                    print(f"    {key}: {value}")

    print("\n" + "=" * 60)

    # Summary
    healthy = sum(1 for s in report.services if s.status == "healthy")
    total = len(report.services)
    print(f"Summary: {healthy}/{total} services healthy")
    print("=" * 60 + "\n")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check health of AutoCognitix services"
    )
    parser.add_argument("--postgres", action="store_true", help="Check only PostgreSQL")
    parser.add_argument("--neo4j", action="store_true", help="Check only Neo4j")
    parser.add_argument("--qdrant", action="store_true", help="Check only Qdrant")
    parser.add_argument("--redis", action="store_true", help="Check only Redis")
    parser.add_argument("--api", action="store_true", help="Check only Backend API")
    parser.add_argument("--api-url", default="http://localhost:8000", help="Backend API URL")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    # If no specific service is selected, check all
    check_all = not any([args.postgres, args.neo4j, args.qdrant, args.redis, args.api])

    checker = HealthChecker()
    report = await checker.run_all_checks(
        check_postgres=args.postgres or check_all,
        check_neo4j=args.neo4j or check_all,
        check_qdrant=args.qdrant or check_all,
        check_redis=args.redis or check_all,
        check_api=args.api or check_all,
        api_url=args.api_url,
    )

    print_report(report, as_json=args.json)

    # Exit with appropriate code
    if report.overall_status == "healthy":
        sys.exit(0)
    elif report.overall_status == "degraded":
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == "__main__":
    asyncio.run(main())
