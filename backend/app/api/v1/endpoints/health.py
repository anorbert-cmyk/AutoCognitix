"""
Health check endpoints for AutoCognitix backend.

Provides detailed health status for all system components:
- PostgreSQL database
- Neo4j graph database
- Qdrant vector database
- Redis cache
- System metrics

Endpoints:
- /health/live - Kubernetes liveness probe (is the app running?)
- /health/ready - Kubernetes readiness probe (is the app ready to serve?)
- /health/detailed - Comprehensive health status for all services
- /health/db - Database-specific statistics
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import text

from app.core.config import settings
from app.core.logging import get_logger
from app.core.metrics import (
    set_data_metrics,
    set_db_pool_metrics,
    update_system_metrics,
)
from app.db.postgres.session import async_session_maker

logger = get_logger(__name__)

router = APIRouter()


# =============================================================================
# Response Models
# =============================================================================


class ServiceHealth(BaseModel):
    """Health status for a single service."""

    name: str
    status: str  # "healthy", "degraded", "unhealthy", "unknown"
    latency_ms: float = 0.0
    details: dict[str, Any] = {}
    error: str | None = None


class DetailedHealthResponse(BaseModel):
    """Detailed health response with all services."""

    status: str
    version: str
    environment: str
    uptime_seconds: float
    services: dict[str, ServiceHealth]
    checked_at: str


class ReadinessResponse(BaseModel):
    """Readiness probe response."""

    status: str
    checks: dict[str, bool]
    checked_at: str


class LivenessResponse(BaseModel):
    """Liveness probe response."""

    status: str
    checked_at: str


class DatabaseStats(BaseModel):
    """Database statistics."""

    postgres: dict[str, Any]
    neo4j: dict[str, Any]
    qdrant: dict[str, Any]
    redis: dict[str, Any]


# Track startup time for uptime calculation
_startup_time: float | None = None


def get_startup_time() -> float:
    """Get or initialize startup time."""
    global _startup_time
    if _startup_time is None:
        _startup_time = time.time()
    return _startup_time


# =============================================================================
# Health Check Functions
# =============================================================================


async def check_postgres_health() -> ServiceHealth:
    """Check PostgreSQL database health."""
    start_time = time.time()
    try:
        async with async_session_maker() as session:
            # Basic connectivity test
            result = await session.execute(text("SELECT 1"))
            result.fetchone()

            # Get table counts for key tables
            table_counts = {}
            for table in ["dtc_codes", "vehicle_makes", "vehicle_models", "users"]:
                try:
                    count_result = await session.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    row = count_result.fetchone()
                    table_counts[table] = row[0] if row else 0
                except Exception:
                    table_counts[table] = "not found"

            # Update metrics
            dtc_count = table_counts.get("dtc_codes", 0)
            if isinstance(dtc_count, int):
                set_data_metrics(dtc_count=dtc_count)

            # Connection pool info (estimated)
            pool_status = {
                "pool_size": 5,
                "max_overflow": 10,
                "status": "active",
            }

            # Update pool metrics
            set_db_pool_metrics("postgres", active=1, idle=4)

        latency = (time.time() - start_time) * 1000

        return ServiceHealth(
            name="PostgreSQL",
            status="healthy",
            latency_ms=round(latency, 2),
            details={
                "table_counts": table_counts,
                "pool_status": pool_status,
            },
        )

    except Exception as e:
        latency = (time.time() - start_time) * 1000
        logger.error(f"PostgreSQL health check failed: {e}")
        return ServiceHealth(
            name="PostgreSQL",
            status="unhealthy",
            latency_ms=round(latency, 2),
            error=str(e),
        )


async def check_neo4j_health() -> ServiceHealth:
    """Check Neo4j database health."""
    from neo4j import GraphDatabase
    from neo4j.exceptions import AuthError, ServiceUnavailable

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

            # Get node counts by label
            node_counts = {}
            for label in ["DTCCode", "Symptom", "Component", "Repair"]:
                try:
                    count_result = session.run(f"MATCH (n:{label}) RETURN COUNT(n) AS count")
                    node_counts[label] = count_result.single()["count"]
                except Exception:
                    node_counts[label] = 0

            # Get relationship count
            rel_result = session.run("MATCH ()-[r]->() RETURN COUNT(r) AS count")
            rel_count = rel_result.single()["count"]

        driver.close()
        latency = (time.time() - start_time) * 1000

        # Update pool metrics
        set_db_pool_metrics("neo4j", active=1, idle=0)

        return ServiceHealth(
            name="Neo4j",
            status="healthy",
            latency_ms=round(latency, 2),
            details={
                "node_counts": node_counts,
                "relationship_count": rel_count,
                "uri": settings.NEO4J_URI.split("@")[-1]
                if "@" in settings.NEO4J_URI
                else settings.NEO4J_URI,
            },
        )

    except AuthError as e:
        latency = (time.time() - start_time) * 1000
        logger.warning(f"Neo4j authentication failed: {e}")
        return ServiceHealth(
            name="Neo4j",
            status="unhealthy",
            latency_ms=round(latency, 2),
            error="Authentication failed",
        )
    except ServiceUnavailable as e:
        latency = (time.time() - start_time) * 1000
        logger.warning(f"Neo4j service unavailable: {e}")
        return ServiceHealth(
            name="Neo4j",
            status="unhealthy",
            latency_ms=round(latency, 2),
            error="Service unavailable",
        )
    except Exception as e:
        latency = (time.time() - start_time) * 1000
        logger.error(f"Neo4j health check failed: {e}")
        return ServiceHealth(
            name="Neo4j",
            status="unhealthy",
            latency_ms=round(latency, 2),
            error=str(e),
        )


async def check_qdrant_health() -> ServiceHealth:
    """Check Qdrant vector database health."""
    from qdrant_client import QdrantClient

    start_time = time.time()
    try:
        # Connect to Qdrant
        if settings.QDRANT_URL:
            client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
            )
            qdrant_location = settings.QDRANT_URL.split("//")[-1].split("/")[0]
        else:
            client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
            )
            qdrant_location = f"{settings.QDRANT_HOST}:{settings.QDRANT_PORT}"

        # Get collections info
        collections = client.get_collections()
        collection_info = {}

        for collection in collections.collections:
            try:
                info = client.get_collection(collection.name)
                collection_info[collection.name] = {
                    "vectors_count": getattr(info, "indexed_vectors_count", getattr(info, "vectors_count", 0)),
                    "points_count": info.points_count,
                    "status": str(info.status),
                }
            except Exception as e:
                collection_info[collection.name] = {"error": str(e)}

        latency = (time.time() - start_time) * 1000

        # Update pool metrics
        set_db_pool_metrics("qdrant", active=1, idle=0)

        return ServiceHealth(
            name="Qdrant",
            status="healthy",
            latency_ms=round(latency, 2),
            details={
                "location": qdrant_location,
                "collections_count": len(collections.collections),
                "collections": collection_info,
            },
        )

    except Exception as e:
        latency = (time.time() - start_time) * 1000
        logger.error(f"Qdrant health check failed: {e}")
        return ServiceHealth(
            name="Qdrant",
            status="unhealthy",
            latency_ms=round(latency, 2),
            error=str(e),
        )


async def check_redis_health() -> ServiceHealth:
    """Check Redis cache health."""
    import redis

    start_time = time.time()
    try:
        client = redis.from_url(settings.REDIS_URL)

        # Basic connectivity test
        pong = client.ping()

        # Get Redis info
        info = client.info()
        memory = client.info("memory")

        latency = (time.time() - start_time) * 1000

        # Update pool metrics
        set_db_pool_metrics(
            "redis",
            active=info.get("connected_clients", 1),
            idle=0,
        )

        return ServiceHealth(
            name="Redis",
            status="healthy" if pong else "degraded",
            latency_ms=round(latency, 2),
            details={
                "version": info.get("redis_version", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": memory.get("used_memory_human", "unknown"),
                "uptime_in_seconds": info.get("uptime_in_seconds", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
            },
        )

    except redis.ConnectionError as e:
        latency = (time.time() - start_time) * 1000
        logger.warning(f"Redis connection failed: {e}")
        return ServiceHealth(
            name="Redis",
            status="unhealthy",
            latency_ms=round(latency, 2),
            error="Connection failed",
        )
    except Exception as e:
        latency = (time.time() - start_time) * 1000
        logger.error(f"Redis health check failed: {e}")
        return ServiceHealth(
            name="Redis",
            status="unhealthy",
            latency_ms=round(latency, 2),
            error=str(e),
        )


# =============================================================================
# API Endpoints
# =============================================================================


@router.get("/live", response_model=LivenessResponse, tags=["Health"])
async def liveness_check():
    """
    Kubernetes/Railway liveness probe.

    Simple check to verify the application is running.
    This endpoint should always return 200 if the process is alive.

    Use this for container orchestration liveness probes.
    If this fails, the container should be restarted.

    Returns:
        LivenessResponse: Status "alive" with timestamp
    """
    return LivenessResponse(
        status="alive",
        checked_at=datetime.utcnow().isoformat() + "Z",
    )


@router.get("/ready", response_model=ReadinessResponse, tags=["Health"])
async def readiness_check():
    """
    Kubernetes/Railway readiness probe.

    Checks if the application is ready to accept traffic.
    Verifies connectivity to critical services (PostgreSQL).

    Use this for container orchestration readiness probes.
    If this fails, traffic should be routed away from this instance.

    Returns:
        ReadinessResponse: Status and individual check results

    Raises:
        HTTPException: 503 if critical services are unavailable
    """
    checks = {}

    # Check PostgreSQL (critical)
    postgres = await check_postgres_health()
    checks["postgres"] = postgres.status == "healthy"

    # Determine overall readiness
    # Only PostgreSQL is critical for readiness
    is_ready = checks["postgres"]

    if not is_ready:
        logger.warning(
            "Readiness check failed",
            extra={
                "event": "readiness_failed",
                "checks": checks,
            },
        )
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "checks": checks,
                "message": "Critical services unavailable",
            },
        )

    return ReadinessResponse(
        status="ready",
        checks=checks,
        checked_at=datetime.utcnow().isoformat() + "Z",
    )


@router.get("/detailed", response_model=DetailedHealthResponse, tags=["Health"])
async def detailed_health_check():
    """
    Detailed health check for all services.

    Returns comprehensive health status including:
    - PostgreSQL database connection and table counts
    - Neo4j graph database node and relationship counts
    - Qdrant vector database collection status
    - Redis cache connection and memory usage
    - Application uptime

    This is for monitoring dashboards and debugging.
    Not recommended for orchestration probes due to slower response.

    Returns:
        DetailedHealthResponse: Complete health status for all services
    """
    # Update system metrics
    update_system_metrics()

    # Initialize startup time if needed
    startup_time = get_startup_time()
    uptime = time.time() - startup_time

    # Run all health checks concurrently
    postgres, neo4j, qdrant, redis_health = await asyncio.gather(
        check_postgres_health(),
        check_neo4j_health(),
        check_qdrant_health(),
        check_redis_health(),
        return_exceptions=True,
    )

    # Handle any exceptions
    services = {}

    if isinstance(postgres, Exception):
        services["postgres"] = ServiceHealth(
            name="PostgreSQL",
            status="unhealthy",
            error=str(postgres),
        )
    else:
        services["postgres"] = postgres

    if isinstance(neo4j, Exception):
        services["neo4j"] = ServiceHealth(
            name="Neo4j",
            status="unhealthy",
            error=str(neo4j),
        )
    else:
        services["neo4j"] = neo4j

    if isinstance(qdrant, Exception):
        services["qdrant"] = ServiceHealth(
            name="Qdrant",
            status="unhealthy",
            error=str(qdrant),
        )
    else:
        services["qdrant"] = qdrant

    if isinstance(redis_health, Exception):
        services["redis"] = ServiceHealth(
            name="Redis",
            status="unhealthy",
            error=str(redis_health),
        )
    else:
        services["redis"] = redis_health

    # Determine overall status
    statuses = [s.status for s in services.values()]
    if all(s == "healthy" for s in statuses):
        overall_status = "healthy"
    elif any(s == "unhealthy" for s in statuses):
        overall_status = "unhealthy"
    elif any(s == "degraded" for s in statuses):
        overall_status = "degraded"
    else:
        overall_status = "unknown"

    logger.info(
        "Detailed health check completed",
        extra={
            "event": "health_check_complete",
            "overall_status": overall_status,
            "services": {name: svc.status for name, svc in services.items()},
            "uptime_seconds": round(uptime, 2),
        },
    )

    return DetailedHealthResponse(
        status=overall_status,
        version="0.1.0",
        environment=settings.ENVIRONMENT,
        uptime_seconds=round(uptime, 2),
        services=services,
        checked_at=datetime.utcnow().isoformat() + "Z",
    )


@router.get("/db", tags=["Health"])
async def database_stats():
    """
    Get database statistics.

    Returns detailed statistics for all databases:
    - PostgreSQL table counts and connection info
    - Neo4j node/relationship counts
    - Qdrant collection stats
    - Redis memory and key counts

    Returns:
        dict: Database statistics for all services
    """
    postgres, neo4j, qdrant, redis_health = await asyncio.gather(
        check_postgres_health(),
        check_neo4j_health(),
        check_qdrant_health(),
        check_redis_health(),
        return_exceptions=True,
    )

    return {
        "postgres": postgres.details
        if not isinstance(postgres, Exception)
        else {"error": str(postgres)},
        "neo4j": neo4j.details if not isinstance(neo4j, Exception) else {"error": str(neo4j)},
        "qdrant": qdrant.details if not isinstance(qdrant, Exception) else {"error": str(qdrant)},
        "redis": redis_health.details
        if not isinstance(redis_health, Exception)
        else {"error": str(redis_health)},
        "checked_at": datetime.utcnow().isoformat() + "Z",
    }
