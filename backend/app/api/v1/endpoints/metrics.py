"""
Prometheus metrics endpoint for AutoCognitix backend.

Provides system metrics in Prometheus format for monitoring:
- Request counts and latencies by endpoint
- Database connection and query metrics
- Embedding generation metrics
- Diagnosis request tracking
- Error rates and system resources

Endpoints:
- /metrics - Prometheus text format metrics for scraping
- /metrics/summary - Human-readable JSON metrics summary
"""


from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import text

from app.core.config import settings
from app.core.logging import get_logger
from app.core.metrics import (
    DTC_CODES_TOTAL,
    get_metrics_summary,
    update_system_metrics,
)

logger = get_logger(__name__)

router = APIRouter()


@router.get("", tags=["Metrics"])
async def get_metrics():
    """
    Prometheus metrics endpoint.

    Returns all metrics in Prometheus text format for scraping by
    Prometheus server or compatible monitoring systems.

    Metrics include:
    - HTTP request count and latency by endpoint
    - Database query times and connection pool status
    - Embedding generation and vector search metrics
    - Diagnosis request tracking
    - Error rates and exception counts
    - System resource usage (CPU, memory)

    Returns:
        Response: Prometheus text format metrics
    """
    # Update system metrics before generating response
    update_system_metrics()

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@router.get("/summary", tags=["Metrics"])
async def get_metrics_summary_endpoint():
    """
    Human-readable metrics summary.

    Returns a JSON summary of key metrics for dashboard display
    and quick status checks.

    Returns:
        dict: Summary of key metrics
    """
    # Get base summary
    summary = get_metrics_summary()

    # Add database metrics
    dtc_count = 0
    try:
        # Use sync engine for simple query
        db_url = settings.DATABASE_URL
        if db_url.startswith("postgresql+asyncpg://"):
            db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")

        from sqlalchemy import create_engine
        engine = create_engine(db_url)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM dtc_codes"))
            row = result.fetchone()
            dtc_count = row[0] if row else 0
    except Exception as e:
        logger.debug(f"Could not fetch DTC count: {e}")

    # Update gauge
    DTC_CODES_TOTAL.set(dtc_count)

    summary["data"] = {
        "dtc_codes_total": dtc_count,
    }

    return summary


@router.get("/prometheus", tags=["Metrics"])
async def get_prometheus_metrics():
    """
    Alias for main metrics endpoint (Prometheus format).

    Some monitoring systems expect /metrics/prometheus endpoint.

    Returns:
        Response: Prometheus text format metrics
    """
    return await get_metrics()
