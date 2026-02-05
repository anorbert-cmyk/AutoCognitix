"""
Redis caching service for AutoCognitix.

Provides high-performance caching for:
- DTC code lookups (most frequently accessed)
- API response caching
- Session data
- Rate limiting counters

Cache Strategies:
- Write-through: Update cache on database writes
- Time-based expiration: Different TTLs per data type
- LRU eviction: Redis handles memory limits
"""

import asyncio
import hashlib
import json
import logging
from functools import wraps
from typing import Any, Callable, List, Optional, TypeVar, Union

import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool

from app.core.config import settings

logger = logging.getLogger(__name__)

# Type variable for generic cache decorator
T = TypeVar("T")

# =============================================================================
# Cache TTL Configuration (in seconds)
# =============================================================================

class CacheTTL:
    """Cache time-to-live configuration per data type."""

    # DTC codes change rarely - cache for 1 hour
    DTC_CODE = 3600

    # DTC search results - cache for 15 minutes
    DTC_SEARCH = 900

    # Known issues - cache for 30 minutes
    KNOWN_ISSUES = 1800

    # Vehicle makes/models - cache for 24 hours (very static)
    VEHICLE_DATA = 86400

    # API response cache - 5 minutes (dynamic data)
    API_RESPONSE = 300

    # NHTSA data - cache for 6 hours (external API)
    NHTSA_DATA = 21600

    # Embedding vectors - cache for 1 hour
    EMBEDDINGS = 3600

    # User session data - cache for 30 minutes
    SESSION = 1800


# =============================================================================
# Cache Key Prefixes
# =============================================================================

class CachePrefix:
    """Cache key prefixes for namespace organization."""

    DTC_CODE = "dtc:code:"
    DTC_SEARCH = "dtc:search:"
    DTC_RELATED = "dtc:related:"
    KNOWN_ISSUES = "issues:"
    VEHICLE_MAKE = "vehicle:make:"
    VEHICLE_MODEL = "vehicle:model:"
    NHTSA_RECALLS = "nhtsa:recalls:"
    NHTSA_COMPLAINTS = "nhtsa:complaints:"
    NHTSA_VIN = "nhtsa:vin:"
    EMBEDDING = "embed:"
    API_RESPONSE = "api:"
    RATE_LIMIT = "ratelimit:"


# =============================================================================
# Redis Cache Service
# =============================================================================

class RedisCacheService:
    """
    Async Redis caching service with connection pooling.

    Features:
    - Connection pooling for high concurrency
    - Automatic serialization/deserialization
    - Batch operations support
    - Circuit breaker pattern for resilience
    - Monitoring and statistics
    """

    _instance: Optional["RedisCacheService"] = None
    _pool: Optional[ConnectionPool] = None

    def __new__(cls) -> "RedisCacheService":
        """Singleton pattern for shared connection pool."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the Redis cache service."""
        if self._initialized:
            return

        self._initialized = True
        self._client: Optional[redis.Redis] = None
        self._connected = False
        self._circuit_open = False
        self._failure_count = 0
        self._max_failures = 5

        logger.info("RedisCacheService initialized")

    async def connect(self) -> None:
        """
        Connect to Redis with connection pooling.

        Connection pool settings:
        - max_connections: Maximum connections in pool
        - socket_timeout: Timeout for operations
        - socket_connect_timeout: Timeout for connection
        - retry_on_timeout: Retry on timeout errors
        """
        if self._connected:
            return

        try:
            # Create connection pool
            RedisCacheService._pool = ConnectionPool.from_url(
                settings.REDIS_URL,
                max_connections=20,
                socket_timeout=5.0,
                socket_connect_timeout=5.0,
                retry_on_timeout=True,
                decode_responses=True,
            )

            # Create client with pool
            self._client = redis.Redis(connection_pool=RedisCacheService._pool)

            # Test connection
            await self._client.ping()
            self._connected = True
            self._circuit_open = False
            self._failure_count = 0

            logger.info(f"Connected to Redis: {settings.REDIS_URL}")

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._connected = False
            raise

    async def disconnect(self) -> None:
        """Disconnect from Redis and close pool."""
        if self._client:
            await self._client.close()
            self._client = None

        if RedisCacheService._pool:
            await RedisCacheService._pool.disconnect()
            RedisCacheService._pool = None

        self._connected = False
        logger.info("Disconnected from Redis")

    async def _check_circuit(self) -> bool:
        """
        Circuit breaker pattern - prevent cascading failures.

        Returns:
            True if circuit is closed (operations allowed), False if open.
        """
        if self._circuit_open:
            # Try to reset after 30 seconds
            return False
        return True

    async def _record_failure(self) -> None:
        """Record a failure and potentially open the circuit."""
        self._failure_count += 1
        if self._failure_count >= self._max_failures:
            self._circuit_open = True
            logger.warning("Redis circuit breaker opened due to failures")

            # Schedule circuit reset after 30 seconds
            asyncio.create_task(self._reset_circuit())

    async def _reset_circuit(self) -> None:
        """Reset the circuit breaker after cooldown period."""
        await asyncio.sleep(30)
        self._circuit_open = False
        self._failure_count = 0
        logger.info("Redis circuit breaker reset")

    # =========================================================================
    # Core Cache Operations
    # =========================================================================

    async def get(self, key: str) -> Optional[Any]:
        """
        Get a value from cache.

        Args:
            key: Cache key.

        Returns:
            Cached value or None if not found/error.
        """
        if not self._connected or not await self._check_circuit():
            return None

        try:
            value = await self._client.get(key)
            if value:
                return json.loads(value)
            return None
        except json.JSONDecodeError:
            return value  # Return raw value if not JSON
        except Exception as e:
            logger.warning(f"Redis GET error for {key}: {e}")
            await self._record_failure()
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int = CacheTTL.API_RESPONSE,
    ) -> bool:
        """
        Set a value in cache with TTL.

        Args:
            key: Cache key.
            value: Value to cache (will be JSON serialized).
            ttl: Time-to-live in seconds.

        Returns:
            True if successful, False otherwise.
        """
        if not self._connected or not await self._check_circuit():
            return False

        try:
            serialized = json.dumps(value, default=str)
            await self._client.setex(key, ttl, serialized)
            return True
        except Exception as e:
            logger.warning(f"Redis SET error for {key}: {e}")
            await self._record_failure()
            return False

    async def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        if not self._connected or not await self._check_circuit():
            return False

        try:
            await self._client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Redis DELETE error for {key}: {e}")
            await self._record_failure()
            return False

    async def delete_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching a pattern.

        Args:
            pattern: Redis pattern (e.g., "dtc:code:*").

        Returns:
            Number of keys deleted.
        """
        if not self._connected or not await self._check_circuit():
            return 0

        try:
            keys = []
            async for key in self._client.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                return await self._client.delete(*keys)
            return 0
        except Exception as e:
            logger.warning(f"Redis DELETE PATTERN error for {pattern}: {e}")
            await self._record_failure()
            return 0

    async def exists(self, key: str) -> bool:
        """Check if a key exists."""
        if not self._connected or not await self._check_circuit():
            return False

        try:
            return await self._client.exists(key) > 0
        except Exception as e:
            logger.warning(f"Redis EXISTS error for {key}: {e}")
            return False

    # =========================================================================
    # Batch Operations
    # =========================================================================

    async def mget(self, keys: List[str]) -> List[Optional[Any]]:
        """
        Get multiple values at once.

        Args:
            keys: List of cache keys.

        Returns:
            List of values (None for missing keys).
        """
        if not self._connected or not await self._check_circuit() or not keys:
            return [None] * len(keys)

        try:
            values = await self._client.mget(keys)
            return [
                json.loads(v) if v else None
                for v in values
            ]
        except Exception as e:
            logger.warning(f"Redis MGET error: {e}")
            await self._record_failure()
            return [None] * len(keys)

    async def mset(
        self,
        mapping: dict,
        ttl: int = CacheTTL.API_RESPONSE,
    ) -> bool:
        """
        Set multiple values at once with same TTL.

        Args:
            mapping: Dictionary of key-value pairs.
            ttl: Time-to-live in seconds.

        Returns:
            True if successful, False otherwise.
        """
        if not self._connected or not await self._check_circuit() or not mapping:
            return False

        try:
            # Use pipeline for atomic batch operation
            async with self._client.pipeline() as pipe:
                for key, value in mapping.items():
                    serialized = json.dumps(value, default=str)
                    pipe.setex(key, ttl, serialized)
                await pipe.execute()
            return True
        except Exception as e:
            logger.warning(f"Redis MSET error: {e}")
            await self._record_failure()
            return False

    # =========================================================================
    # DTC Code Caching
    # =========================================================================

    async def get_dtc_code(self, code: str) -> Optional[dict]:
        """Get a DTC code from cache."""
        key = f"{CachePrefix.DTC_CODE}{code.upper()}"
        return await self.get(key)

    async def set_dtc_code(self, code: str, data: dict) -> bool:
        """Cache a DTC code."""
        key = f"{CachePrefix.DTC_CODE}{code.upper()}"
        return await self.set(key, data, CacheTTL.DTC_CODE)

    async def get_dtc_search_results(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 20,
    ) -> Optional[List[dict]]:
        """Get cached DTC search results."""
        key = self._make_search_key(query, category, limit)
        return await self.get(key)

    async def set_dtc_search_results(
        self,
        query: str,
        results: List[dict],
        category: Optional[str] = None,
        limit: int = 20,
    ) -> bool:
        """Cache DTC search results."""
        key = self._make_search_key(query, category, limit)
        return await self.set(key, results, CacheTTL.DTC_SEARCH)

    def _make_search_key(
        self,
        query: str,
        category: Optional[str],
        limit: int,
    ) -> str:
        """Generate a consistent cache key for search queries."""
        params = f"{query.lower()}:{category or 'all'}:{limit}"
        hash_val = hashlib.md5(params.encode()).hexdigest()[:12]
        return f"{CachePrefix.DTC_SEARCH}{hash_val}"

    async def get_related_codes(self, code: str) -> Optional[List[dict]]:
        """Get cached related DTC codes."""
        key = f"{CachePrefix.DTC_RELATED}{code.upper()}"
        return await self.get(key)

    async def set_related_codes(self, code: str, related: List[dict]) -> bool:
        """Cache related DTC codes."""
        key = f"{CachePrefix.DTC_RELATED}{code.upper()}"
        return await self.set(key, related, CacheTTL.DTC_CODE)

    # =========================================================================
    # NHTSA Data Caching
    # =========================================================================

    async def get_nhtsa_recalls(
        self,
        make: str,
        model: str,
        year: int,
    ) -> Optional[List[dict]]:
        """Get cached NHTSA recalls."""
        key = f"{CachePrefix.NHTSA_RECALLS}{make}:{model}:{year}"
        return await self.get(key)

    async def set_nhtsa_recalls(
        self,
        make: str,
        model: str,
        year: int,
        recalls: List[dict],
    ) -> bool:
        """Cache NHTSA recalls."""
        key = f"{CachePrefix.NHTSA_RECALLS}{make}:{model}:{year}"
        return await self.set(key, recalls, CacheTTL.NHTSA_DATA)

    async def get_nhtsa_complaints(
        self,
        make: str,
        model: str,
        year: int,
    ) -> Optional[List[dict]]:
        """Get cached NHTSA complaints."""
        key = f"{CachePrefix.NHTSA_COMPLAINTS}{make}:{model}:{year}"
        return await self.get(key)

    async def set_nhtsa_complaints(
        self,
        make: str,
        model: str,
        year: int,
        complaints: List[dict],
    ) -> bool:
        """Cache NHTSA complaints."""
        key = f"{CachePrefix.NHTSA_COMPLAINTS}{make}:{model}:{year}"
        return await self.set(key, complaints, CacheTTL.NHTSA_DATA)

    async def get_vin_decode(self, vin: str) -> Optional[dict]:
        """Get cached VIN decode result."""
        key = f"{CachePrefix.NHTSA_VIN}{vin.upper()}"
        return await self.get(key)

    async def set_vin_decode(self, vin: str, data: dict) -> bool:
        """Cache VIN decode result."""
        key = f"{CachePrefix.NHTSA_VIN}{vin.upper()}"
        return await self.set(key, data, CacheTTL.NHTSA_DATA)

    # =========================================================================
    # Embedding Caching
    # =========================================================================

    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding vector."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        key = f"{CachePrefix.EMBEDDING}{text_hash}"
        return await self.get(key)

    async def set_embedding(self, text: str, embedding: List[float]) -> bool:
        """Cache embedding vector."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        key = f"{CachePrefix.EMBEDDING}{text_hash}"
        return await self.set(key, embedding, CacheTTL.EMBEDDINGS)

    async def get_embeddings_batch(
        self,
        texts: List[str],
    ) -> List[Optional[List[float]]]:
        """Get multiple cached embeddings."""
        keys = [
            f"{CachePrefix.EMBEDDING}{hashlib.md5(t.encode()).hexdigest()}"
            for t in texts
        ]
        return await self.mget(keys)

    # =========================================================================
    # Rate Limiting
    # =========================================================================

    async def check_rate_limit(
        self,
        identifier: str,
        limit: int,
        window_seconds: int,
    ) -> tuple[bool, int]:
        """
        Check and increment rate limit counter.

        Args:
            identifier: Unique identifier (e.g., IP address, user ID).
            limit: Maximum requests allowed in window.
            window_seconds: Time window in seconds.

        Returns:
            Tuple of (allowed: bool, remaining: int).
        """
        if not self._connected or not await self._check_circuit():
            return True, limit  # Allow if Redis unavailable

        key = f"{CachePrefix.RATE_LIMIT}{identifier}"

        try:
            async with self._client.pipeline() as pipe:
                pipe.incr(key)
                pipe.expire(key, window_seconds)
                results = await pipe.execute()

            current = results[0]
            remaining = max(0, limit - current)
            allowed = current <= limit

            return allowed, remaining

        except Exception as e:
            logger.warning(f"Rate limit check error: {e}")
            return True, limit

    # =========================================================================
    # Statistics
    # =========================================================================

    async def get_stats(self) -> dict:
        """Get cache statistics."""
        if not self._connected:
            return {"status": "disconnected"}

        try:
            info = await self._client.info()
            return {
                "status": "connected",
                "circuit_open": self._circuit_open,
                "used_memory": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "total_keys": await self._client.dbsize(),
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "hit_rate": self._calculate_hit_rate(info),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _calculate_hit_rate(self, info: dict) -> float:
        """Calculate cache hit rate percentage."""
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        total = hits + misses
        if total == 0:
            return 0.0
        return round((hits / total) * 100, 2)


# =============================================================================
# Global Instance
# =============================================================================

_cache_service: Optional[RedisCacheService] = None


async def get_cache_service() -> RedisCacheService:
    """
    Get the global cache service instance.

    Ensures connection is established.
    """
    global _cache_service
    if _cache_service is None:
        _cache_service = RedisCacheService()
        await _cache_service.connect()
    return _cache_service


# =============================================================================
# Cache Decorator
# =============================================================================

def cached(
    prefix: str,
    ttl: int = CacheTTL.API_RESPONSE,
    key_builder: Optional[Callable[..., str]] = None,
):
    """
    Decorator for caching function results.

    Args:
        prefix: Cache key prefix.
        ttl: Time-to-live in seconds.
        key_builder: Optional function to build cache key from args.

    Usage:
        @cached(CachePrefix.DTC_CODE, CacheTTL.DTC_CODE)
        async def get_dtc_code(code: str) -> dict:
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            # Build cache key
            if key_builder:
                key = f"{prefix}{key_builder(*args, **kwargs)}"
            else:
                # Default: hash all arguments
                arg_str = f"{args}:{kwargs}"
                key = f"{prefix}{hashlib.md5(arg_str.encode()).hexdigest()[:16]}"

            # Try to get from cache
            try:
                cache = await get_cache_service()
                cached_value = await cache.get(key)
                if cached_value is not None:
                    logger.debug(f"Cache HIT: {key}")
                    return cached_value
            except Exception:
                pass  # Cache miss or error - continue to function

            # Execute function
            result = await func(*args, **kwargs)

            # Store in cache
            try:
                cache = await get_cache_service()
                await cache.set(key, result, ttl)
                logger.debug(f"Cache SET: {key}")
            except Exception:
                pass  # Don't fail if caching fails

            return result

        return wrapper
    return decorator
