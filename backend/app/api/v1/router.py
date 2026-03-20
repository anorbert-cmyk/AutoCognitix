"""
API v1 router - aggregates all endpoint routers.
"""

from fastapi import APIRouter

from app.api.v1.endpoints import (
    auth,
    calculator,
    chat,
    diagnosis,
    dtc_codes,
    health,
    inspection,
    metrics,
    services,
    vehicles,
)

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(
    auth.router,
    prefix="/auth",
    tags=["Authentication"],
)

api_router.include_router(
    diagnosis.router,
    prefix="/diagnosis",
    tags=["Diagnosis"],
)

api_router.include_router(
    vehicles.router,
    prefix="/vehicles",
    tags=["Vehicles"],
)

api_router.include_router(
    dtc_codes.router,
    prefix="/dtc",
    tags=["DTC Codes"],
)

api_router.include_router(
    health.router,
    prefix="/health",
    tags=["Health"],
)

api_router.include_router(
    metrics.router,
    prefix="/metrics",
    tags=["Metrics"],
)

api_router.include_router(
    inspection.router,
    prefix="/inspection",
    tags=["Inspection"],
)

api_router.include_router(
    calculator.router,
    prefix="/calculator",
    tags=["Calculator"],
)

api_router.include_router(
    chat.router,
    prefix="/chat",
    tags=["Chat"],
)

api_router.include_router(
    services.router,
    prefix="/services",
    tags=["Services"],
)
