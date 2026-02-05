"""
Retry utilities with exponential backoff for external API calls.

Provides:
- Configurable retry decorator
- Exponential backoff with jitter
- Specific exception handling
- Logging of retry attempts
"""

import asyncio
import random
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from app.core.exceptions import (
    LLMException,
    LLMRateLimitException,
    NHTSAException,
    NHTSARateLimitException,
)
from app.core.logging import get_logger

logger = get_logger(__name__)

# Type variable for generic function return
T = TypeVar("T")


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        jitter_factor: float = 0.1,
        retryable_exceptions: tuple[type[Exception], ...] | None = None,
        retryable_status_codes: tuple[int, ...] | None = None,
    ):
        """
        Initialize retry configuration.

        Args:
            max_attempts: Maximum number of retry attempts
            base_delay: Initial delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            exponential_base: Base for exponential backoff calculation
            jitter: Whether to add random jitter to delay
            jitter_factor: Factor for jitter calculation (0.0-1.0)
            retryable_exceptions: Tuple of exception types to retry on
            retryable_status_codes: Tuple of HTTP status codes to retry on
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.jitter_factor = jitter_factor
        self.retryable_exceptions = retryable_exceptions or (
            ConnectionError,
            TimeoutError,
            asyncio.TimeoutError,
        )
        self.retryable_status_codes = retryable_status_codes or (
            429,  # Too Many Requests
            500,  # Internal Server Error
            502,  # Bad Gateway
            503,  # Service Unavailable
            504,  # Gateway Timeout
        )


# Default configurations for different services
DEFAULT_CONFIG = RetryConfig()

NHTSA_CONFIG = RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    max_delay=30.0,
    retryable_exceptions=(
        ConnectionError,
        TimeoutError,
        asyncio.TimeoutError,
        NHTSAException,
    ),
)

LLM_CONFIG = RetryConfig(
    max_attempts=3,
    base_delay=2.0,
    max_delay=60.0,
    retryable_exceptions=(
        ConnectionError,
        TimeoutError,
        asyncio.TimeoutError,
        LLMException,
    ),
)


def calculate_delay(
    attempt: int,
    config: RetryConfig,
    retry_after: int | None = None,
) -> float:
    """
    Calculate delay for next retry attempt.

    Args:
        attempt: Current attempt number (0-indexed)
        config: Retry configuration
        retry_after: Optional retry-after header value

    Returns:
        Delay in seconds
    """
    # Use retry-after header if provided
    if retry_after is not None:
        return min(retry_after, config.max_delay)

    # Calculate exponential backoff
    delay = config.base_delay * (config.exponential_base ** attempt)

    # Add jitter if enabled
    if config.jitter:
        jitter_range = delay * config.jitter_factor
        delay += random.uniform(-jitter_range, jitter_range)

    # Clamp to max delay
    return min(delay, config.max_delay)


def is_retryable_exception(
    exception: Exception,
    config: RetryConfig,
) -> bool:
    """Check if exception should be retried."""
    return isinstance(exception, config.retryable_exceptions)


def is_rate_limit_exception(exception: Exception) -> tuple[bool, int | None]:
    """
    Check if exception is a rate limit error and extract retry-after.

    Returns:
        Tuple of (is_rate_limit, retry_after_seconds)
    """
    if isinstance(exception, (NHTSARateLimitException, LLMRateLimitException)):
        return True, getattr(exception, "retry_after", 60)

    # Check for httpx response with 429 status
    if hasattr(exception, "response"):
        response = exception.response
        if hasattr(response, "status_code") and response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    return True, int(retry_after)
                except ValueError:
                    pass
            return True, 60

    return False, None


def retry_async(
    config: RetryConfig | None = None,
    on_retry: Callable[[Exception, int], None] | None = None,
):
    """
    Decorator for async functions with retry logic.

    Args:
        config: Retry configuration (uses DEFAULT_CONFIG if not provided)
        on_retry: Optional callback called on each retry with (exception, attempt)

    Usage:
        @retry_async(config=LLM_CONFIG)
        async def call_llm(prompt: str) -> str:
            ...

        @retry_async()
        async def fetch_data():
            ...
    """
    retry_config = config or DEFAULT_CONFIG

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Exception | None = None

            for attempt in range(retry_config.max_attempts):
                try:
                    return await func(*args, **kwargs)

                except Exception as e:
                    last_exception = e

                    # Check if this is a rate limit error
                    is_rate_limit, retry_after = is_rate_limit_exception(e)

                    # Check if we should retry
                    if not is_retryable_exception(e, retry_config) and not is_rate_limit:
                        logger.warning(
                            f"Non-retryable exception in {func.__name__}",
                            extra={
                                "function": func.__name__,
                                "attempt": attempt + 1,
                                "error_type": type(e).__name__,
                                "error_message": str(e),
                            },
                        )
                        raise

                    # Check if we've exhausted retries
                    if attempt + 1 >= retry_config.max_attempts:
                        logger.error(
                            f"Max retries exceeded for {func.__name__}",
                            extra={
                                "function": func.__name__,
                                "max_attempts": retry_config.max_attempts,
                                "error_type": type(e).__name__,
                                "error_message": str(e),
                            },
                        )
                        raise

                    # Calculate delay
                    delay = calculate_delay(attempt, retry_config, retry_after)

                    # Log retry
                    logger.warning(
                        f"Retrying {func.__name__} after error",
                        extra={
                            "function": func.__name__,
                            "attempt": attempt + 1,
                            "max_attempts": retry_config.max_attempts,
                            "delay_seconds": round(delay, 2),
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                            "is_rate_limit": is_rate_limit,
                        },
                    )

                    # Call retry callback if provided
                    if on_retry:
                        on_retry(e, attempt)

                    # Wait before retry
                    await asyncio.sleep(delay)

            # Should never reach here, but just in case
            if last_exception:
                raise last_exception

        return wrapper

    return decorator


def retry_sync(
    config: RetryConfig | None = None,
    on_retry: Callable[[Exception, int], None] | None = None,
):
    """
    Decorator for sync functions with retry logic.

    Args:
        config: Retry configuration (uses DEFAULT_CONFIG if not provided)
        on_retry: Optional callback called on each retry with (exception, attempt)

    Usage:
        @retry_sync(config=DEFAULT_CONFIG)
        def fetch_data() -> dict:
            ...
    """
    import time

    retry_config = config or DEFAULT_CONFIG

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Exception | None = None

            for attempt in range(retry_config.max_attempts):
                try:
                    return func(*args, **kwargs)

                except Exception as e:
                    last_exception = e

                    # Check if this is a rate limit error
                    is_rate_limit, retry_after = is_rate_limit_exception(e)

                    # Check if we should retry
                    if not is_retryable_exception(e, retry_config) and not is_rate_limit:
                        logger.warning(
                            f"Non-retryable exception in {func.__name__}",
                            extra={
                                "function": func.__name__,
                                "attempt": attempt + 1,
                                "error_type": type(e).__name__,
                                "error_message": str(e),
                            },
                        )
                        raise

                    # Check if we've exhausted retries
                    if attempt + 1 >= retry_config.max_attempts:
                        logger.error(
                            f"Max retries exceeded for {func.__name__}",
                            extra={
                                "function": func.__name__,
                                "max_attempts": retry_config.max_attempts,
                                "error_type": type(e).__name__,
                                "error_message": str(e),
                            },
                        )
                        raise

                    # Calculate delay
                    delay = calculate_delay(attempt, retry_config, retry_after)

                    # Log retry
                    logger.warning(
                        f"Retrying {func.__name__} after error",
                        extra={
                            "function": func.__name__,
                            "attempt": attempt + 1,
                            "max_attempts": retry_config.max_attempts,
                            "delay_seconds": round(delay, 2),
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                            "is_rate_limit": is_rate_limit,
                        },
                    )

                    # Call retry callback if provided
                    if on_retry:
                        on_retry(e, attempt)

                    # Wait before retry
                    time.sleep(delay)

            # Should never reach here, but just in case
            if last_exception:
                raise last_exception

        return wrapper

    return decorator


class RetryContext:
    """
    Context manager for retry logic with manual control.

    Usage:
        async with RetryContext(config=LLM_CONFIG) as ctx:
            while ctx.should_retry:
                try:
                    result = await call_api()
                    break
                except Exception as e:
                    await ctx.handle_exception(e)
    """

    def __init__(self, config: RetryConfig | None = None):
        self.config = config or DEFAULT_CONFIG
        self.attempt = 0
        self.last_exception: Exception | None = None

    @property
    def should_retry(self) -> bool:
        """Check if another retry should be attempted."""
        return self.attempt < self.config.max_attempts

    async def handle_exception(self, exception: Exception) -> None:
        """
        Handle an exception during retry.

        Args:
            exception: The exception that occurred

        Raises:
            The exception if it's not retryable or max retries exceeded
        """
        self.last_exception = exception
        self.attempt += 1

        # Check if this is a rate limit error
        is_rate_limit, retry_after = is_rate_limit_exception(exception)

        # Check if we should retry
        if not is_retryable_exception(exception, self.config) and not is_rate_limit:
            raise exception

        # Check if we've exhausted retries
        if not self.should_retry:
            raise exception

        # Calculate and apply delay
        delay = calculate_delay(self.attempt - 1, self.config, retry_after)

        logger.warning(
            f"Retry attempt {self.attempt}/{self.config.max_attempts}",
            extra={
                "attempt": self.attempt,
                "max_attempts": self.config.max_attempts,
                "delay_seconds": round(delay, 2),
                "error_type": type(exception).__name__,
                "is_rate_limit": is_rate_limit,
            },
        )

        await asyncio.sleep(delay)

    async def __aenter__(self) -> "RetryContext":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        # Don't suppress exceptions
        return False
