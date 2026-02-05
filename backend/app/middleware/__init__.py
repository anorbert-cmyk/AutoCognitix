"""
Middleware package for AutoCognitix.

Provides:
- MetricsMiddleware: Prometheus-compatible request metrics
- Additional middleware components for request processing
"""

from app.middleware.metrics import (
    MetricsMiddleware,
    get_metrics_middleware,
    track_request,
)

__all__ = [
    "MetricsMiddleware",
    "get_metrics_middleware",
    "track_request",
]
