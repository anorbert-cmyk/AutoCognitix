"""
AutoCognitix - AI-powered Vehicle Diagnostic Platform
Main FastAPI Application Entry Point
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import ORJSONResponse

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.csrf import CSRFMiddleware
from app.core.error_handlers import setup_all_exception_handlers
from app.core.etag import CacheControlMiddleware, ETagMiddleware
from app.core.logging import RequestLoggingMiddleware, get_logger, setup_logging
from app.core.metrics import MetricsMiddleware
from app.core.rate_limiter import RateLimitMiddleware
from app.db.postgres.session import async_session_maker, engine
from app.db.qdrant_client import qdrant_client

logger = get_logger(__name__)

# European vehicle manufacturers seed data
_VEHICLE_MAKES = [
    {"id": "volkswagen", "name": "Volkswagen", "country": "Germany"},
    {"id": "bmw", "name": "BMW", "country": "Germany"},
    {"id": "mercedes", "name": "Mercedes-Benz", "country": "Germany"},
    {"id": "audi", "name": "Audi", "country": "Germany"},
    {"id": "porsche", "name": "Porsche", "country": "Germany"},
    {"id": "opel", "name": "Opel", "country": "Germany"},
    {"id": "skoda", "name": "Skoda", "country": "Czech Republic"},
    {"id": "seat", "name": "SEAT", "country": "Spain"},
    {"id": "renault", "name": "Renault", "country": "France"},
    {"id": "peugeot", "name": "Peugeot", "country": "France"},
    {"id": "citroen", "name": "Citroen", "country": "France"},
    {"id": "fiat", "name": "Fiat", "country": "Italy"},
    {"id": "alfa_romeo", "name": "Alfa Romeo", "country": "Italy"},
    {"id": "ferrari", "name": "Ferrari", "country": "Italy"},
    {"id": "lamborghini", "name": "Lamborghini", "country": "Italy"},
    {"id": "volvo", "name": "Volvo", "country": "Sweden"},
    {"id": "saab", "name": "Saab", "country": "Sweden"},
    {"id": "jaguar", "name": "Jaguar", "country": "United Kingdom"},
    {"id": "land_rover", "name": "Land Rover", "country": "United Kingdom"},
    {"id": "mini", "name": "MINI", "country": "United Kingdom"},
]

_VEHICLE_MODELS = [
    {"id": "vw_golf", "name": "Golf", "make_id": "volkswagen", "year_start": 1974},
    {"id": "vw_passat", "name": "Passat", "make_id": "volkswagen", "year_start": 1973},
    {"id": "vw_tiguan", "name": "Tiguan", "make_id": "volkswagen", "year_start": 2007},
    {"id": "vw_polo", "name": "Polo", "make_id": "volkswagen", "year_start": 1975},
    {"id": "bmw_3series", "name": "3 Series", "make_id": "bmw", "year_start": 1975},
    {"id": "bmw_5series", "name": "5 Series", "make_id": "bmw", "year_start": 1972},
    {"id": "bmw_x3", "name": "X3", "make_id": "bmw", "year_start": 2003},
    {"id": "bmw_x5", "name": "X5", "make_id": "bmw", "year_start": 1999},
    {"id": "mb_cclass", "name": "C-Class", "make_id": "mercedes", "year_start": 1993},
    {"id": "mb_eclass", "name": "E-Class", "make_id": "mercedes", "year_start": 1993},
    {"id": "mb_glc", "name": "GLC", "make_id": "mercedes", "year_start": 2015},
    {"id": "audi_a3", "name": "A3", "make_id": "audi", "year_start": 1996},
    {"id": "audi_a4", "name": "A4", "make_id": "audi", "year_start": 1994},
    {"id": "audi_a6", "name": "A6", "make_id": "audi", "year_start": 1994},
    {"id": "audi_q5", "name": "Q5", "make_id": "audi", "year_start": 2008},
    {"id": "skoda_octavia", "name": "Octavia", "make_id": "skoda", "year_start": 1996},
    {"id": "skoda_fabia", "name": "Fabia", "make_id": "skoda", "year_start": 1999},
    {"id": "skoda_superb", "name": "Superb", "make_id": "skoda", "year_start": 2001},
    {"id": "renault_clio", "name": "Clio", "make_id": "renault", "year_start": 1990},
    {"id": "renault_megane", "name": "Megane", "make_id": "renault", "year_start": 1995},
    {"id": "peugeot_208", "name": "208", "make_id": "peugeot", "year_start": 2012},
    {"id": "peugeot_308", "name": "308", "make_id": "peugeot", "year_start": 2007},
    {"id": "peugeot_3008", "name": "3008", "make_id": "peugeot", "year_start": 2009},
    {"id": "citroen_c3", "name": "C3", "make_id": "citroen", "year_start": 2002},
    {"id": "citroen_c4", "name": "C4", "make_id": "citroen", "year_start": 2004},
    {"id": "fiat_500", "name": "500", "make_id": "fiat", "year_start": 2007},
    {"id": "fiat_panda", "name": "Panda", "make_id": "fiat", "year_start": 1980},
    {"id": "volvo_xc60", "name": "XC60", "make_id": "volvo", "year_start": 2008},
    {"id": "volvo_xc90", "name": "XC90", "make_id": "volvo", "year_start": 2002},
    {"id": "volvo_v60", "name": "V60", "make_id": "volvo", "year_start": 2010},
    {"id": "opel_astra", "name": "Astra", "make_id": "opel", "year_start": 1991},
    {"id": "opel_corsa", "name": "Corsa", "make_id": "opel", "year_start": 1982},
]


async def _seed_vehicle_data() -> None:
    """Seed vehicle makes and models if tables are empty. Idempotent."""
    from sqlalchemy import text

    async with async_session_maker() as session:
        result = await session.execute(text("SELECT COUNT(*) FROM vehicle_makes"))
        count = result.scalar()
        if count and count > 0:
            logger.info(f"Vehicle makes already seeded ({count} records)")
            return

        for make in _VEHICLE_MAKES:
            await session.execute(
                text(
                    "INSERT INTO vehicle_makes (id, name, country) VALUES (:id, :name, :country) "
                    "ON CONFLICT (id) DO NOTHING"
                ),
                make,
            )

        for model in _VEHICLE_MODELS:
            await session.execute(
                text(
                    "INSERT INTO vehicle_models (id, name, make_id, year_start) "
                    "VALUES (:id, :name, :make_id, :year_start) "
                    "ON CONFLICT (id) DO NOTHING"
                ),
                model,
            )

        await session.commit()
        logger.info(f"Seeded {len(_VEHICLE_MAKES)} vehicle makes and {len(_VEHICLE_MODELS)} models")


async def _seed_dtc_codes() -> None:
    """Seed DTC codes from JSON file if table is empty. Idempotent."""
    import json
    import time
    from pathlib import Path

    from sqlalchemy import text

    async with async_session_maker() as session:
        result = await session.execute(text("SELECT COUNT(*) FROM dtc_codes"))
        count = result.scalar()
        if count and count > 0:
            logger.info(f"DTC codes already seeded ({count} records)")
            return

        # Look for seed data file
        seed_file = Path("/app/data/dtc_codes_seed.json")
        if not seed_file.exists():
            # Try relative path for local development
            seed_file = (
                Path(__file__).parent.parent.parent / "data" / "dtc_codes" / "all_codes_merged.json"
            )
        if not seed_file.exists():
            logger.warning("DTC seed data file not found, skipping")
            return

        try:
            with seed_file.open() as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to read DTC seed file: {e}")
            return

        codes = data.get("codes", []) if isinstance(data, dict) else data
        if not codes:
            logger.warning("No DTC codes found in seed file")
            return

        logger.info(f"Seeding {len(codes)} DTC codes...")
        start = time.time()

        # Batch insert using executemany for performance
        insert_sql = text(
            "INSERT INTO dtc_codes (code, description_en, description_hu, category, severity, "
            "is_generic, system, symptoms, possible_causes, diagnostic_steps, related_codes, sources) "
            "VALUES (:code, :description_en, :description_hu, :category, :severity, "
            ":is_generic, :system, :symptoms, :possible_causes, :diagnostic_steps, "
            ":related_codes, :sources) "
            "ON CONFLICT (code) DO NOTHING"
        )

        params = [
            {
                "code": c.get("code", ""),
                "description_en": c.get("description_en", ""),
                "description_hu": c.get("description_hu"),
                "category": c.get("category", "powertrain"),
                "severity": c.get("severity", "medium"),
                "is_generic": c.get("is_generic", True),
                "system": c.get("system"),
                "symptoms": c.get("symptoms", []),
                "possible_causes": c.get("possible_causes", []),
                "diagnostic_steps": c.get("diagnostic_steps", []),
                "related_codes": c.get("related_codes", []),
                "sources": c.get("sources", []),
            }
            for c in codes
        ]

        # Insert in batches of 500 for memory efficiency
        batch_size = 500
        for i in range(0, len(params), batch_size):
            batch = params[i : i + batch_size]
            for p in batch:
                await session.execute(insert_sql, p)

        await session.commit()
        elapsed = time.time() - start
        logger.info(f"Seeded {len(params)} DTC codes in {elapsed:.1f}s")


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

    # Seed vehicle makes/models if tables are empty
    try:
        await _seed_vehicle_data()
    except Exception as e:
        logger.warning(f"Vehicle data seeding skipped: {e}")

    # Seed DTC codes if table is empty
    try:
        await _seed_dtc_codes()
    except Exception as e:
        logger.warning(f"DTC codes seeding skipped: {e}")

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
        allow_headers=[
            "Authorization",
            "Content-Type",
            "Accept",
            "Origin",
            "X-Requested-With",
            "X-Request-ID",
            "X-CSRF-Token",
        ],
        expose_headers=["X-Request-ID", "X-Correlation-ID"],
    )

    # Security headers middleware - protect against common web vulnerabilities
    @application.middleware("http")
    async def add_security_headers(request: Request, call_next):
        """Add security headers to all responses."""
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        if not settings.DEBUG:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response

    # CSRF protection middleware - double-submit cookie pattern
    # Exclude health checks, metrics, and docs from CSRF validation
    application.add_middleware(
        CSRFMiddleware,
        exclude_paths=[
            "/health",
            "/metrics",
            "/api/v1/metrics",
            "/api/v1/docs",
            "/api/v1/redoc",
            "/api/v1/openapi.json",
        ],
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
