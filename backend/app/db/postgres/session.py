"""
PostgreSQL database session configuration with enhanced error handling.

Optimized for performance with:
- Connection pooling (QueuePool with optimized settings)
- Statement caching (prepared_statement_cache_size)
- Pool pre-ping for stale connection detection
- Optimized pool recycling for long-running connections
- Comprehensive error handling with Hungarian error messages
"""

from typing import Any

from sqlalchemy.exc import (
    DBAPIError,
    IntegrityError,
    OperationalError,
    SQLAlchemyError,
)
from sqlalchemy.exc import (
    TimeoutError as SQLAlchemyTimeoutError,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings
from app.core.exceptions import (
    PostgresConnectionException,
    PostgresException,
)
from app.core.logging import get_logger
from collections.abc import AsyncGenerator

logger = get_logger(__name__)

# =============================================================================
# Connection Pool Configuration
# =============================================================================
# Pool settings optimized for typical web workloads:
# - pool_size: Number of permanent connections (baseline)
# - max_overflow: Additional connections during peak load
# - pool_recycle: Recycle connections after N seconds (prevents stale connections)
# - pool_timeout: Wait time before raising TimeoutError
# - pool_pre_ping: Test connections before use (detect stale connections)

POOL_SIZE = int((settings.DEBUG and 5) or 5)  # Conservative for Railway (2 workers x 5 = 10)
MAX_OVERFLOW = int((settings.DEBUG and 10) or 10)  # Max 15 per worker, 30 total
POOL_RECYCLE = 1800  # Recycle connections every 30 minutes
POOL_TIMEOUT = 30  # 30 second timeout for acquiring connection

# =============================================================================
# Engine Configuration
# =============================================================================

# Connection arguments for asyncpg (PostgreSQL async driver)
connect_args = {
    # Enable statement caching for repeated queries (significant speedup)
    "prepared_statement_cache_size": 100,
    # Connection timeout
    "command_timeout": 60,
    # Server settings for performance
    "server_settings": {
        # Work memory for sorts/hashes (per operation)
        "work_mem": "16MB",
        # Enable parallel query execution
        "max_parallel_workers_per_gather": "2",
        # JIT compilation for complex queries
        "jit": "on",
    },
}

# Create async engine with connection pooling
# SQLAlchemy 2.0 async engines support AsyncAdaptedQueuePool by default
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    future=True,
    # Connection pooling settings
    pool_size=POOL_SIZE,
    max_overflow=MAX_OVERFLOW,
    pool_recycle=POOL_RECYCLE,
    pool_timeout=POOL_TIMEOUT,
    pool_pre_ping=True,
    # asyncpg specific settings
    connect_args=connect_args,
    # Execution options
    execution_options={
        "isolation_level": "READ COMMITTED",
    },
)

logger.info(f"Database engine initialized with pool_size={POOL_SIZE}, max_overflow={MAX_OVERFLOW}")

# Create async session factory with optimized settings
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Don't expire objects after commit (reduces queries)
    autocommit=False,
    autoflush=False,  # Manual flush control for batch operations
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency that provides a database session with comprehensive error handling.

    Optimized session handling:
    - Uses connection pool for efficient connection reuse
    - Proper transaction handling with rollback on error
    - Automatic session cleanup
    - Hungarian error messages for client consumption

    Usage:
        @app.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db)):
            ...

    Raises:
        PostgresConnectionException: When unable to connect to database
        PostgresException: For other database errors
    """
    session: AsyncSession | None = None
    try:
        session = async_session_maker()
        yield session
        await session.commit()

    except OperationalError as e:
        # Connection-related errors (connection refused, timeout, etc.)
        logger.error(
            "PostgreSQL connection error",
            extra={
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
            exc_info=True,
        )
        if session:
            await session.rollback()
        raise PostgresConnectionException(
            message="Nem sikerult csatlakozni az adatbazishoz.",
            original_error=e,
        )

    except IntegrityError as e:
        # Constraint violations
        logger.warning(
            "PostgreSQL integrity error",
            extra={
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
        )
        if session:
            await session.rollback()
        raise PostgresException(
            message="Adatintegritasi hiba tortent.",
            details={"constraint_violation": True},
            original_error=e,
        )

    except SQLAlchemyTimeoutError as e:
        # Query timeout
        logger.error(
            "PostgreSQL timeout error",
            extra={
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
        )
        if session:
            await session.rollback()
        raise PostgresException(
            message="Az adatbazis muvelet idotullpest szenvedett.",
            details={"timeout": True},
            original_error=e,
        )

    except DBAPIError as e:
        # Low-level database API errors
        logger.error(
            "PostgreSQL DBAPI error",
            extra={
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
            exc_info=True,
        )
        if session:
            await session.rollback()
        raise PostgresException(
            message="Adatbazis hiba tortent.",
            original_error=e,
        )

    except SQLAlchemyError as e:
        # Generic SQLAlchemy errors
        logger.error(
            "SQLAlchemy error",
            extra={
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
            exc_info=True,
        )
        if session:
            await session.rollback()
        raise PostgresException(
            message="Adatbazis hiba tortent.",
            original_error=e,
        )

    except Exception as e:
        # Unexpected errors - rollback and re-raise
        logger.error(
            "Unexpected database session error",
            extra={
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
            exc_info=True,
        )
        if session:
            await session.rollback()
        raise

    finally:
        if session:
            await session.close()


# =============================================================================
# Pool Statistics (for monitoring)
# =============================================================================


async def get_pool_status() -> dict[str, Any]:
    """
    Get current connection pool status.

    Returns:
        Dictionary with pool statistics for monitoring.
    """
    pool: Any = engine.pool
    return {
        "pool_size": pool.size(),
        "checked_in": pool.checkedin(),
        "checked_out": pool.checkedout(),
        "overflow": pool.overflow(),
        "invalid": pool.invalidatedcount() if hasattr(pool, "invalidatedcount") else 0,
    }


async def dispose_engine() -> None:
    """
    Dispose of the engine and all connections.

    Call this during application shutdown.
    """
    await engine.dispose()
    logger.info("Database engine disposed")


async def check_database_connection() -> bool:
    """
    Check if database connection is available.

    Returns:
        True if connection is available, False otherwise.
    """
    try:
        async with async_session_maker() as session:
            from sqlalchemy import text

            await session.execute(text("SELECT 1"))
            return True
    except Exception as e:
        logger.warning(f"Database connection check failed: {e}")
        return False


async def get_database_info() -> dict[str, Any]:
    """
    Get database connection information for health checks.

    Returns:
        Dictionary with database status and info.
    """
    try:
        async with async_session_maker() as session:
            from sqlalchemy import text

            result = await session.execute(text("SELECT version()"))
            version = result.scalar()

            pool: Any = engine.pool
            return {
                "status": "connected",
                "version": version,
                "pool_size": pool.size(),
                "pool_checkedin": pool.checkedin(),
                "pool_overflow": pool.overflow(),
                "pool_checkedout": pool.checkedout(),
            }
    except Exception as e:
        logger.error(f"Failed to get database info: {e}")
        return {
            "status": "disconnected",
            "error": str(e),
        }
