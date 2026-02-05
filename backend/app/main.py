"""
AutoCognitix - AI-powered Vehicle Diagnostic Platform
Main FastAPI Application Entry Point
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import ORJSONResponse

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.logging import setup_logging, get_logger, RequestLoggingMiddleware
from app.core.metrics import MetricsMiddleware
from app.core.error_handlers import setup_all_exception_handlers
from app.core.rate_limiter import RateLimitMiddleware
from app.core.etag import ETagMiddleware, CacheControlMiddleware
from app.db.postgres.session import engine
from app.db.qdrant_client import qdrant_client

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan handler for startup and shutdown events."""
    # Startup
    setup_logging()
    logger.info("Starting AutoCognitix backend service")

    # Initialize Qdrant collections (skip if not configured for cloud deployment)
    try:
        await qdrant_client.initialize_collections()
        logger.info("Qdrant collections initialized successfully")
    except Exception as e:
        logger.warning(f"Qdrant initialization skipped: {e}")

    # Initialize Redis cache
    try:
        from app.db.redis_cache import get_cache_service
        cache = await get_cache_service()
        stats = await cache.get_stats()
        logger.info(f"Redis cache initialized: {stats.get('status', 'unknown')}")
    except Exception as e:
        logger.warning(f"Redis cache initialization skipped: {e}")

    yield

    # Shutdown
    logger.info("Shutting down AutoCognitix backend service")

    # Close Redis connection
    try:
        from app.db.redis_cache import _cache_service
        if _cache_service:
            await _cache_service.disconnect()
            logger.info("Redis cache connection closed")
    except Exception as e:
        logger.warning(f"Redis cache disconnect error: {e}")

    await engine.dispose()
    logger.info("Database connections closed")


def create_application() -> FastAPI:
    """Create and configure the FastAPI application."""

    # OpenAPI tags metadata for documentation
    tags_metadata = [
        {
            "name": "Health",
            "description": "Service health monitoring and readiness probes. Use these endpoints to verify service availability and check database connectivity.",
        },
        {
            "name": "Authentication",
            "description": "User authentication and authorization. Register new users, obtain JWT tokens, and manage sessions. All authenticated endpoints require a Bearer token.",
        },
        {
            "name": "Diagnosis",
            "description": "Core vehicle diagnostic functionality. Analyze DTC codes and symptoms using AI-powered RAG system with Hungarian language support.",
        },
        {
            "name": "DTC Codes",
            "description": "Diagnostic Trouble Code (DTC) lookup and search. Access the OBD-II code database with Hungarian translations and detailed repair information.",
        },
        {
            "name": "Vehicles",
            "description": "Vehicle information management. Decode VINs, lookup vehicle makes/models, and retrieve NHTSA recall and complaint data.",
        },
        {
            "name": "Metrics",
            "description": "Prometheus metrics for monitoring. Track request counts, latencies, and application health metrics.",
        },
    ]

    application = FastAPI(
        title=settings.PROJECT_NAME,
        description="""
# AutoCognitix API

AI-powered Vehicle Diagnostic Platform with Hungarian language support.

## Features

- **AI-Powered Diagnostics**: Analyze DTC codes and symptoms using RAG (Retrieval Augmented Generation)
- **Hungarian Language Support**: Full Hungarian NLP with huBERT embeddings
- **OBD-II Database**: Comprehensive DTC code database with translations
- **VIN Decoding**: Decode Vehicle Identification Numbers using NHTSA API
- **NHTSA Integration**: Access recalls and complaints data

## Authentication

Most endpoints require JWT authentication. Obtain tokens via `/api/v1/auth/login`.

```
Authorization: Bearer <access_token>
```

## Rate Limits

- 60 requests per minute
- 1000 requests per hour

## Support

For API support, visit the [project repository](https://github.com/autocognitix).
        """,
        version="0.1.0",
        openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
        docs_url=f"{settings.API_V1_PREFIX}/docs",
        redoc_url=f"{settings.API_V1_PREFIX}/redoc",
        default_response_class=ORJSONResponse,
        lifespan=lifespan,
        openapi_tags=tags_metadata,
        contact={
            "name": "AutoCognitix Support",
            "email": "support@autocognitix.hu",
        },
        license_info={
            "name": "MIT License",
            "url": "https://opensource.org/licenses/MIT",
        },
    )

    # GZip compression middleware - compress responses > 1KB
    # Provides significant bandwidth savings for API responses
    application.add_middleware(GZipMiddleware, minimum_size=1000)

    # ETag middleware for HTTP caching (after GZip to hash compressed content)
    application.add_middleware(
        ETagMiddleware,
        exclude_paths=["/api/v1/metrics", "/health", "/api/v1/auth"],
    )

    # Cache-Control headers middleware
    application.add_middleware(CacheControlMiddleware)

    # Rate limiting middleware (must be added first to process before other middleware)
    application.add_middleware(RateLimitMiddleware)

    # Metrics collection middleware (collects request metrics for Prometheus)
    application.add_middleware(MetricsMiddleware)

    # Request logging middleware (must be added before CORS)
    application.add_middleware(RequestLoggingMiddleware)

    # CORS middleware - Restricted to specific methods and headers for security
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.BACKEND_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "Accept", "Origin", "X-Requested-With", "X-Request-ID"],
        expose_headers=["X-Request-ID", "X-Correlation-ID"],
    )

    # Setup exception handlers
    setup_all_exception_handlers(application)

    # Include API router
    application.include_router(api_router, prefix=settings.API_V1_PREFIX)

    # Root health check endpoint for container orchestration
    @application.get("/health", tags=["Health"])
    async def health_check():
        """Basic health check endpoint for container orchestration."""
        return {
            "status": "healthy",
            "version": "0.1.0",
            "service": "autocognitix-backend",
            "environment": settings.ENVIRONMENT,
        }

    # Liveness probe at root level
    @application.get("/health/live", tags=["Health"])
    async def health_live():
        """
        Kubernetes liveness probe - verifies the application is running.
        For full details, use /api/v1/health/live
        """
        from app.api.v1.endpoints.health import liveness_check
        return await liveness_check()

    # Readiness probe at root level
    @application.get("/health/ready", tags=["Health"])
    async def health_ready():
        """
        Kubernetes readiness probe - verifies the application can accept traffic.
        For full details, use /api/v1/health/ready
        """
        from app.api.v1.endpoints.health import readiness_check
        return await readiness_check()

    # Detailed health endpoint at root level (redirects to API v1)
    @application.get("/health/detailed", tags=["Health"])
    async def health_detailed():
        """
        Detailed health check - redirects to API v1 health endpoint.
        For full details, use /api/v1/health/detailed
        """
        from app.api.v1.endpoints.health import detailed_health_check
        return await detailed_health_check()

    # Database stats at root level
    @application.get("/health/db", tags=["Health"])
    async def health_db():
        """
        Database statistics - redirects to API v1 health endpoint.
        For full details, use /api/v1/health/db
        """
        from app.api.v1.endpoints.health import database_stats
        return await database_stats()

    # Metrics endpoint at root level
    @application.get("/metrics", tags=["Metrics"])
    async def metrics():
        """
        Prometheus metrics endpoint.
        For full details, use /api/v1/metrics
        """
        from app.api.v1.endpoints.metrics import get_metrics
        return await get_metrics()

    return application


app = create_application()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
