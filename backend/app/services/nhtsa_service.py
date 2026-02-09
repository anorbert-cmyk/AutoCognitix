"""
NHTSA (National Highway Traffic Safety Administration) API Client Service.

Provides async methods to interact with NHTSA APIs for:
- VIN decoding
- Vehicle recalls
- Vehicle complaints

Features:
- Async HTTP client with connection pooling
- Automatic retry with exponential backoff
- Rate limiting support
- Optional Redis caching
"""

import asyncio
import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, cast

import httpx
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings
from app.core.log_sanitizer import sanitize_log
from app.core.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Pydantic Models
# =============================================================================


class VINDecodeResult(BaseModel):
    """Result model for VIN decoding."""

    vin: str = Field(..., description="The decoded VIN")
    make: Optional[str] = Field(None, description="Vehicle manufacturer")
    model: Optional[str] = Field(None, description="Vehicle model")
    model_year: Optional[int] = Field(None, description="Model year")
    body_class: Optional[str] = Field(None, description="Body class/type")
    vehicle_type: Optional[str] = Field(None, description="Vehicle type")
    plant_city: Optional[str] = Field(None, description="Manufacturing plant city")
    plant_country: Optional[str] = Field(None, description="Manufacturing plant country")
    manufacturer: Optional[str] = Field(None, description="Full manufacturer name")
    engine_cylinders: Optional[int] = Field(None, description="Number of engine cylinders")
    engine_displacement_l: Optional[float] = Field(
        None, description="Engine displacement in liters"
    )
    fuel_type_primary: Optional[str] = Field(None, description="Primary fuel type")
    transmission_style: Optional[str] = Field(None, description="Transmission style")
    drive_type: Optional[str] = Field(None, description="Drive type (FWD, RWD, AWD, etc.)")
    doors: Optional[int] = Field(None, description="Number of doors")
    gvwr: Optional[str] = Field(None, description="Gross Vehicle Weight Rating")
    error_code: Optional[str] = Field(None, description="Error code if decoding failed")
    error_text: Optional[str] = Field(None, description="Error description if decoding failed")
    raw_data: Dict[str, Any] = Field(default_factory=dict, description="Complete raw API response")

    @property
    def is_valid(self) -> bool:
        """Check if VIN decode was successful."""
        return self.error_code in (None, "0", "")


class Recall(BaseModel):
    """Model for vehicle recall information."""

    campaign_number: str = Field(..., description="NHTSA campaign number")
    manufacturer: str = Field(..., description="Vehicle manufacturer")
    make: str = Field(..., description="Vehicle make")
    model: str = Field(..., description="Vehicle model")
    model_year: int = Field(..., description="Model year")
    recall_date: Optional[str] = Field(None, description="Date recall was announced")
    component: str = Field(..., description="Affected component")
    summary: str = Field(..., description="Brief summary of the recall")
    consequence: Optional[str] = Field(None, description="Potential consequence of the defect")
    remedy: Optional[str] = Field(None, description="Manufacturer's remedy")
    notes: Optional[str] = Field(None, description="Additional notes")
    nhtsa_id: Optional[str] = Field(None, description="NHTSA ID")


class Complaint(BaseModel):
    """Model for vehicle complaint information."""

    odinumber: Optional[str] = Field(None, description="ODI number")
    manufacturer: str = Field(..., description="Vehicle manufacturer")
    make: str = Field(..., description="Vehicle make")
    model: str = Field(..., description="Vehicle model")
    model_year: int = Field(..., description="Model year")
    crash: bool = Field(False, description="Whether a crash occurred")
    fire: bool = Field(False, description="Whether a fire occurred")
    injuries: int = Field(0, description="Number of injuries reported")
    deaths: int = Field(0, description="Number of deaths reported")
    complaint_date: Optional[str] = Field(None, description="Date complaint was filed")
    date_of_incident: Optional[str] = Field(None, description="Date of the incident")
    components: Optional[str] = Field(None, description="Affected components")
    summary: Optional[str] = Field(None, description="Complaint description/summary")


class NHTSAError(Exception):
    """Custom exception for NHTSA API errors."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class RateLimitError(NHTSAError):
    """Exception raised when rate limit is exceeded."""

    pass


# =============================================================================
# Cache Implementation
# =============================================================================


class CacheBackend:
    """Abstract base for cache backends."""

    async def get(self, key: str) -> Optional[str]:
        raise NotImplementedError

    async def set(self, key: str, value: str, ttl: int = 3600) -> None:
        raise NotImplementedError

    async def delete(self, key: str) -> None:
        raise NotImplementedError


class InMemoryCache(CacheBackend):
    """Simple in-memory cache implementation."""

    def __init__(self):
        self._cache: Dict[str, Tuple[str, float]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[str]:
        async with self._lock:
            if key in self._cache:
                value, expiry = self._cache[key]
                if expiry > datetime.now().timestamp():
                    return value
                del self._cache[key]
            return None

    async def set(self, key: str, value: str, ttl: int = 3600) -> None:
        async with self._lock:
            expiry = datetime.now().timestamp() + ttl
            self._cache[key] = (value, expiry)

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._cache.pop(key, None)

    async def clear_expired(self) -> int:
        """Clear expired entries and return count of removed items."""
        async with self._lock:
            now = datetime.now().timestamp()
            expired_keys = [k for k, (_, exp) in self._cache.items() if exp <= now]
            for key in expired_keys:
                del self._cache[key]
            return len(expired_keys)


class RedisCache(CacheBackend):
    """Redis-based cache implementation."""

    def __init__(self, redis_url: str):
        self._redis_url = redis_url
        self._redis = None

    async def _get_redis(self):
        """Lazy initialization of Redis connection."""
        if self._redis is None:
            try:
                import redis.asyncio as aioredis

                self._redis = await aioredis.from_url(
                    self._redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                )
            except ImportError:
                logger.warning("redis package not installed, falling back to in-memory cache")
                raise
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise
        return self._redis

    async def get(self, key: str) -> Optional[str]:
        try:
            redis = await self._get_redis()
            result = await redis.get(key)
            return str(result) if result is not None else None
        except Exception as e:
            logger.warning(f"Redis get failed: {e}")
            return None

    async def set(self, key: str, value: str, ttl: int = 3600) -> None:
        try:
            redis = await self._get_redis()
            await redis.setex(key, ttl, value)
        except Exception as e:
            logger.warning(f"Redis set failed: {e}")

    async def delete(self, key: str) -> None:
        try:
            redis = await self._get_redis()
            await redis.delete(key)
        except Exception as e:
            logger.warning(f"Redis delete failed: {e}")

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None


# =============================================================================
# NHTSA Service
# =============================================================================


class NHTSAService:
    """
    Async client for NHTSA APIs.

    Provides methods to:
    - Decode VIN numbers
    - Fetch recall information
    - Fetch complaint information

    Features:
    - Automatic retry with exponential backoff
    - Connection pooling
    - Optional caching (Redis or in-memory)
    - Rate limiting awareness
    """

    # API base URLs
    VPIC_BASE_URL = "https://vpic.nhtsa.dot.gov/api/vehicles"
    RECALLS_BASE_URL = "https://api.nhtsa.gov/recalls"
    COMPLAINTS_BASE_URL = "https://api.nhtsa.gov/complaints"

    # Rate limiting settings
    REQUESTS_PER_SECOND = 5
    RATE_LIMIT_WINDOW = 1.0  # seconds

    # Cache TTL settings (in seconds)
    VIN_CACHE_TTL = 86400  # 24 hours
    RECALLS_CACHE_TTL = 3600  # 1 hour
    COMPLAINTS_CACHE_TTL = 3600  # 1 hour

    def __init__(
        self,
        use_redis: bool = False,
        redis_url: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """
        Initialize NHTSA service.

        Args:
            use_redis: Whether to use Redis for caching (default: False)
            redis_url: Redis connection URL (uses settings.REDIS_URL if not provided)
            timeout: HTTP request timeout in seconds
        """
        self._client: httpx.AsyncClient | None = None
        self._timeout = timeout
        self._use_redis = use_redis
        self._redis_url = redis_url or settings.REDIS_URL
        self._cache: CacheBackend | None = None
        self._request_timestamps: list[float] = []
        self._rate_limit_lock = asyncio.Lock()

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout),
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
                headers={
                    "User-Agent": f"AutoCognitix/{settings.PROJECT_NAME}",
                    "Accept": "application/json",
                },
            )
        return self._client

    async def _get_cache(self) -> CacheBackend:
        """Get or create cache backend."""
        if self._cache is None:
            if self._use_redis:
                try:
                    self._cache = RedisCache(self._redis_url)
                    # Test connection
                    await self._cache.get("__test__")
                    logger.info("Using Redis cache for NHTSA service")
                except Exception as e:
                    logger.warning(f"Redis unavailable, falling back to in-memory cache: {e}")
                    self._cache = InMemoryCache()
            else:
                self._cache = InMemoryCache()
                logger.info("Using in-memory cache for NHTSA service")
        return self._cache

    def _generate_cache_key(self, prefix: str, *args) -> str:
        """Generate a cache key from prefix and arguments."""
        key_data = f"{prefix}:{':'.join(str(a).lower() for a in args)}"
        return f"nhtsa:{hashlib.md5(key_data.encode()).hexdigest()}"

    async def _check_rate_limit(self) -> None:
        """Check and enforce rate limiting."""
        async with self._rate_limit_lock:
            now = datetime.now().timestamp()
            # Remove timestamps older than the rate limit window
            self._request_timestamps = [
                ts for ts in self._request_timestamps if now - ts < self.RATE_LIMIT_WINDOW
            ]
            # Check if we've exceeded the rate limit
            if len(self._request_timestamps) >= self.REQUESTS_PER_SECOND:
                sleep_time = self.RATE_LIMIT_WINDOW - (now - self._request_timestamps[0])
                if sleep_time > 0:
                    logger.debug(f"Rate limit reached, sleeping for {sleep_time:.2f}s")
                    await asyncio.sleep(sleep_time)
            # Record this request
            self._request_timestamps.append(datetime.now().timestamp())

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _make_request(
        self,
        method: str,
        url: str,
        params: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Make HTTP request with retry logic.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            params: Query parameters

        Returns:
            JSON response as dictionary

        Raises:
            NHTSAError: If request fails after retries
            RateLimitError: If rate limit is exceeded
        """
        await self._check_rate_limit()
        client = await self._get_client()

        try:
            logger.debug(
                f"Making {method} request to {sanitize_log(url)} with params {sanitize_log(str(params))}"
            )
            response = await client.request(method, url, params=params)

            # Check for rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 60))
                logger.warning(f"Rate limited by NHTSA, retry after {retry_after}s")
                raise RateLimitError(
                    f"Rate limit exceeded, retry after {retry_after} seconds",
                    status_code=429,
                )

            response.raise_for_status()
            return cast("Dict[str, Any]", response.json())

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from NHTSA: {e.response.status_code} - {e}")
            raise NHTSAError(
                f"HTTP error: {e.response.status_code}",
                status_code=e.response.status_code,
            )
        except httpx.TimeoutException as e:
            logger.error(f"Timeout error from NHTSA: {e}")
            raise
        except httpx.HTTPError as e:
            logger.error(f"HTTP error from NHTSA: {e}")
            raise

    # =========================================================================
    # VIN Decoding
    # =========================================================================

    async def decode_vin(self, vin: str, use_cache: bool = True) -> VINDecodeResult:
        """
        Decode a VIN (Vehicle Identification Number).

        Args:
            vin: 17-character VIN to decode
            use_cache: Whether to use cached results (default: True)

        Returns:
            VINDecodeResult with vehicle information

        Raises:
            NHTSAError: If API request fails
            ValueError: If VIN format is invalid
        """
        # Validate VIN format
        vin = vin.strip().upper()
        if len(vin) != 17:
            raise ValueError(f"VIN must be 17 characters, got {len(vin)}")

        # Check cache
        cache_key = self._generate_cache_key("vin", vin)
        if use_cache:
            cache = await self._get_cache()
            cached = await cache.get(cache_key)
            if cached:
                logger.debug(f"Cache hit for VIN {sanitize_log(vin)}")
                return VINDecodeResult(**json.loads(cached))

        # Make API request
        url = f"{self.VPIC_BASE_URL}/DecodeVinValues/{vin}"
        params = {"format": "json"}

        try:
            data = await self._make_request("GET", url, params)
            results = data.get("Results", [{}])[0]

            # Parse response
            result = VINDecodeResult(
                vin=vin,
                make=results.get("Make") or None,
                model=results.get("Model") or None,
                model_year=self._safe_int(results.get("ModelYear")),
                body_class=results.get("BodyClass") or None,
                vehicle_type=results.get("VehicleType") or None,
                plant_city=results.get("PlantCity") or None,
                plant_country=results.get("PlantCountry") or None,
                manufacturer=results.get("Manufacturer") or None,
                engine_cylinders=self._safe_int(results.get("EngineCylinders")),
                engine_displacement_l=self._safe_float(results.get("DisplacementL")),
                fuel_type_primary=results.get("FuelTypePrimary") or None,
                transmission_style=results.get("TransmissionStyle") or None,
                drive_type=results.get("DriveType") or None,
                doors=self._safe_int(results.get("Doors")),
                gvwr=results.get("GVWR") or None,
                error_code=results.get("ErrorCode") or None,
                error_text=results.get("ErrorText") or None,
                raw_data=results,
            )

            # Cache successful results
            if use_cache and result.is_valid:
                await cache.set(cache_key, result.model_dump_json(), self.VIN_CACHE_TTL)

            logger.info(
                f"Decoded VIN {sanitize_log(vin)}: {sanitize_log(result.make)} {sanitize_log(result.model)} {result.model_year}"
            )
            return result

        except Exception as e:
            logger.error(f"Failed to decode VIN {sanitize_log(vin)}: {sanitize_log(str(e))}")
            raise

    # =========================================================================
    # Recalls
    # =========================================================================

    async def get_recalls(
        self,
        make: str,
        model: str,
        year: int,
        use_cache: bool = True,
    ) -> List[Recall]:
        """
        Get recall information for a vehicle.

        Args:
            make: Vehicle make (e.g., "Toyota")
            model: Vehicle model (e.g., "Camry")
            year: Model year
            use_cache: Whether to use cached results (default: True)

        Returns:
            List of Recall objects

        Raises:
            NHTSAError: If API request fails
        """
        make = make.strip()
        model = model.strip()

        # Check cache
        cache_key = self._generate_cache_key("recalls", make, model, year)
        if use_cache:
            cache = await self._get_cache()
            cached = await cache.get(cache_key)
            if cached:
                logger.debug(
                    f"Cache hit for recalls: {sanitize_log(make)} {sanitize_log(model)} {year}"
                )
                return [Recall(**r) for r in json.loads(cached)]

        # Make API request
        url = f"{self.RECALLS_BASE_URL}/recallsByVehicle"
        params = {
            "make": make,
            "model": model,
            "modelYear": year,
        }

        try:
            data = await self._make_request("GET", url, params)
            results = data.get("results", [])

            recalls = []
            for item in results:
                recall = Recall(
                    campaign_number=item.get("NHTSACampaignNumber", ""),
                    manufacturer=item.get("Manufacturer", make),
                    make=make,
                    model=model,
                    model_year=year,
                    recall_date=item.get("ReportReceivedDate"),
                    component=item.get("Component", "Unknown"),
                    summary=item.get("Summary", ""),
                    consequence=item.get("Consequence"),
                    remedy=item.get("Remedy"),
                    notes=item.get("Notes"),
                    nhtsa_id=item.get("NHTSAActionNumber"),
                )
                recalls.append(recall)

            # Cache results
            if use_cache:
                await cache.set(
                    cache_key,
                    json.dumps([r.model_dump() for r in recalls]),
                    self.RECALLS_CACHE_TTL,
                )

            logger.info(
                f"Found {len(recalls)} recalls for {sanitize_log(make)} {sanitize_log(model)} {year}"
            )
            return recalls

        except Exception as e:
            logger.error(
                f"Failed to get recalls for {sanitize_log(make)} {sanitize_log(model)} {year}: {sanitize_log(str(e))}"
            )
            raise

    # =========================================================================
    # Complaints
    # =========================================================================

    async def get_complaints(
        self,
        make: str,
        model: str,
        year: int,
        use_cache: bool = True,
    ) -> List[Complaint]:
        """
        Get complaint information for a vehicle.

        Args:
            make: Vehicle make (e.g., "Toyota")
            model: Vehicle model (e.g., "Camry")
            year: Model year
            use_cache: Whether to use cached results (default: True)

        Returns:
            List of Complaint objects

        Raises:
            NHTSAError: If API request fails
        """
        make = make.strip()
        model = model.strip()

        # Check cache
        cache_key = self._generate_cache_key("complaints", make, model, year)
        if use_cache:
            cache = await self._get_cache()
            cached = await cache.get(cache_key)
            if cached:
                logger.debug(
                    f"Cache hit for complaints: {sanitize_log(make)} {sanitize_log(model)} {year}"
                )
                return [Complaint(**c) for c in json.loads(cached)]

        # Make API request
        url = f"{self.COMPLAINTS_BASE_URL}/complaintsByVehicle"
        params = {
            "make": make,
            "model": model,
            "modelYear": year,
        }

        try:
            data = await self._make_request("GET", url, params)
            results = data.get("results", [])

            complaints = []
            for item in results:
                odi = item.get("odiNumber")
                complaint = Complaint(
                    odinumber=str(odi) if odi is not None else None,
                    manufacturer=item.get("manufacturer", make),
                    make=make,
                    model=model,
                    model_year=year,
                    crash=item.get("crash", "N") == "Y",
                    fire=item.get("fire", "N") == "Y",
                    injuries=self._safe_int(item.get("numberOfInjuries")) or 0,
                    deaths=self._safe_int(item.get("numberOfDeaths")) or 0,
                    complaint_date=item.get("dateComplaintFiled"),
                    date_of_incident=item.get("dateOfIncident"),
                    components=item.get("components"),
                    summary=item.get("summary"),
                )
                complaints.append(complaint)

            # Cache results
            if use_cache:
                await cache.set(
                    cache_key,
                    json.dumps([c.model_dump() for c in complaints]),
                    self.COMPLAINTS_CACHE_TTL,
                )

            logger.info(
                f"Found {len(complaints)} complaints for {sanitize_log(make)} {sanitize_log(model)} {year}"
            )
            return complaints

        except Exception as e:
            logger.error(
                f"Failed to get complaints for {sanitize_log(make)} {sanitize_log(model)} {year}: {sanitize_log(str(e))}"
            )
            raise

    # =========================================================================
    # Utility Methods
    # =========================================================================

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        """Safely convert value to int."""
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        """Safely convert value to float."""
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    async def close(self) -> None:
        """Close HTTP client and cache connections."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

        if self._cache:
            if isinstance(self._cache, RedisCache):
                await self._cache.close()
            self._cache = None

        logger.info("NHTSA service closed")

    async def __aenter__(self) -> "NHTSAService":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()


# =============================================================================
# Service Instance Factory
# =============================================================================


_service_instance: NHTSAService | None = None


async def get_nhtsa_service(use_redis: bool = False) -> NHTSAService:
    """
    Get or create NHTSA service instance.

    Args:
        use_redis: Whether to use Redis for caching

    Returns:
        NHTSAService instance
    """
    global _service_instance
    if _service_instance is None:
        _service_instance = NHTSAService(use_redis=use_redis)
    return _service_instance


async def close_nhtsa_service() -> None:
    """Close the global NHTSA service instance."""
    global _service_instance
    if _service_instance is not None:
        await _service_instance.close()
        _service_instance = None
