"""
Diagnosis endpoints - core functionality for vehicle diagnostics.

Provides AI-powered vehicle diagnosis using:
- DTC code analysis
- Hungarian symptom text processing with huBERT
- RAG-based knowledge retrieval from Neo4j graph database
- Vector similarity search with Qdrant
- Streaming SSE support for real-time AI output
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.endpoints.auth import get_current_user_from_token, get_optional_current_user
from app.api.v1.schemas.diagnosis import (
    DeleteResponse,
    DiagnosisHistoryItem,
    DiagnosisRequest,
    DiagnosisResponse,
    DiagnosisStats,
    DiagnosisStreamRequest,
    DTCFrequency,
    MonthlyDiagnosisCount,
    PaginatedDiagnosisHistory,
    StreamingEvent,
    VehicleDiagnosisCount,
)
from app.core.exceptions import (
    AutoCognitixException,
    DiagnosisException,
    DTCValidationException,
    NotFoundException,
    VINValidationException,
)
from app.core.log_sanitizer import sanitize_exception, sanitize_log
from app.core.logging import get_logger
from app.db.postgres.models import User
from app.db.postgres.repositories import DiagnosisSessionRepository
from app.db.postgres.session import get_db
from app.services.diagnosis_service import (
    DiagnosisService,
    DiagnosisServiceError,
    DTCValidationError,
    VINDecodeError,
)

router = APIRouter()
logger = get_logger(__name__)

# Streaming endpoint protection constants
STREAM_TIMEOUT_SECONDS = 300  # 5-minute max duration per stream
MAX_CONCURRENT_STREAMS = 10  # Limit concurrent long-running streams
_stream_semaphore = asyncio.Semaphore(MAX_CONCURRENT_STREAMS)

# OpenAPI response examples
ANALYZE_RESPONSES: Dict[Union[int, str], Dict[str, Any]] = {
    201: {
        "description": "Diagnosis completed successfully",
        "content": {
            "application/json": {
                "example": {
                    "id": "550e8400-e29b-41d4-a716-446655440000",
                    "vehicle_make": "Volkswagen",
                    "vehicle_model": "Golf",
                    "vehicle_year": 2018,
                    "dtc_codes": ["P0101", "P0171"],
                    "symptoms": "A motor nehezen indul hidegben...",
                    "probable_causes": [
                        {
                            "title": "MAF szenzor hiba",
                            "description": "A levegotomeg-mero szenzor hibas vagy szennyezett.",
                            "confidence": 0.85,
                            "related_dtc_codes": ["P0101"],
                            "components": ["MAF szenzor", "Levegoszuro"],
                        }
                    ],
                    "recommended_repairs": [
                        {
                            "title": "MAF szenzor tisztitasa/csereje",
                            "description": "Ellenorizze es tisztitsa meg a MAF szenzort specialis tisztitoval.",
                            "estimated_cost_min": 5000,
                            "estimated_cost_max": 45000,
                            "estimated_cost_currency": "HUF",
                            "difficulty": "intermediate",
                            "parts_needed": ["MAF szenzor tisztito"],
                            "estimated_time_minutes": 30,
                        }
                    ],
                    "confidence_score": 0.82,
                    "sources": [
                        {
                            "type": "database",
                            "title": "OBD-II DTC Database",
                            "url": None,
                            "relevance_score": 0.95,
                        }
                    ],
                    "created_at": "2024-02-03T10:30:00Z",
                }
            }
        },
    },
    400: {
        "description": "Invalid request data",
        "content": {
            "application/json": {
                "examples": {
                    "invalid_dtc": {
                        "summary": "Invalid DTC code",
                        "value": {"detail": "Invalid DTC code format: X1234"},
                    },
                    "invalid_vin": {
                        "summary": "Invalid VIN",
                        "value": {"detail": "Invalid VIN: VIN must be exactly 17 characters"},
                    },
                }
            }
        },
    },
    500: {
        "description": "Internal server error",
        "content": {
            "application/json": {
                "example": {"detail": "An error occurred during diagnosis. Please try again."}
            }
        },
    },
}

QUICK_ANALYZE_RESPONSES: Dict[Union[int, str], Dict[str, Any]] = {
    200: {
        "description": "Quick analysis completed",
        "content": {
            "application/json": {
                "example": {
                    "dtc_codes": [
                        {
                            "code": "P0101",
                            "description": "Levegotomeg-mero aramkor tartomany/teljesitmeny hiba",
                            "severity": "medium",
                            "symptoms": ["Motor teljesitmenyvesztese", "Egyenetlen alapjarat"],
                            "possible_causes": ["Szennyezett MAF szenzor", "Levegoszuro eltomodes"],
                        }
                    ],
                    "message": "Reszletes diagnozishoz hasznalja a /analyze vegpontot jarmuadatokkal.",
                }
            }
        },
    },
    400: {
        "description": "Invalid DTC code format",
        "content": {"application/json": {"example": {"detail": "Invalid DTC code format: X1234"}}},
    },
}


@router.post(
    "/analyze",
    response_model=DiagnosisResponse,
    status_code=status.HTTP_201_CREATED,
    responses=ANALYZE_RESPONSES,
    summary="Analyze vehicle (main diagnostic endpoint)",
    description="""
**Main diagnostic endpoint** - Analyze vehicle based on DTC codes and symptoms.

This endpoint performs comprehensive AI-powered diagnosis:

1. **DTC Code Processing**: Validates and looks up diagnostic trouble codes
2. **Hungarian NLP**: Processes symptom text using huBERT embeddings
3. **Vector Search**: Finds similar issues in Qdrant vector database
4. **Graph Query**: Traverses Neo4j knowledge graph for diagnostic paths
5. **AI Generation**: Produces diagnosis with confidence scores using LLM

**Request Body:**
- `vehicle_make`, `vehicle_model`, `vehicle_year`: Vehicle identification (required)
- `dtc_codes`: Array of DTC codes like P0101, B1234 (1-20 codes)
- `symptoms`: Symptom description in Hungarian (10-2000 characters)
- `vin`: Optional 17-character VIN for additional vehicle data
- `additional_context`: Optional extra context

**Response includes:**
- Ranked probable causes with confidence scores
- Repair recommendations with cost estimates (HUF)
- Related information sources
- Overall confidence score

**Hungarian language example:**
```json
{
  "symptoms": "A motor nehezen indul hidegben, egyenetlenul jar alapjaraton."
}
```
    """,
)
async def analyze_vehicle(
    request: DiagnosisRequest,
    response: Response,
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_current_user),
):
    """
    Analyze vehicle based on DTC codes and symptoms.

    This is the main diagnostic endpoint that:
    1. Processes DTC codes and Hungarian symptom text
    2. Searches vector database for similar issues
    3. Queries knowledge graph for diagnostic paths
    4. Generates AI-powered diagnosis with confidence scores

    Args:
        request: Diagnosis request with vehicle info, DTCs, and symptoms
        db: Database session
        current_user: Optional authenticated user (diagnosis is saved to user history if authenticated)

    Returns:
        Comprehensive diagnosis with probable causes and repair recommendations

    Raises:
        400: Invalid DTC codes or VIN
        500: Internal service error
    """
    try:
        user_id = UUID(str(current_user.id)) if current_user else None
        async with DiagnosisService(db) as service:
            result = await service.analyze_vehicle(request, user_id=user_id)
            response.headers["Location"] = f"/api/v1/diagnosis/{result.id}"

            # Check if this was a duplicate submission
            duplicate_of = getattr(result, "_duplicate_of", None)
            if duplicate_of is not None:
                return JSONResponse(
                    content=result.model_dump(mode="json"),
                    status_code=status.HTTP_201_CREATED,
                    headers={"X-Duplicate-Of": str(duplicate_of)},
                )

            return result

    except DTCValidationError as e:
        logger.warning(
            "DTC validation error",
            extra={
                "error_message": sanitize_exception(e),
                "dtc_codes": sanitize_log(request.dtc_codes),
            },
        )
        raise DTCValidationException(
            message=str(e),
            invalid_codes=request.dtc_codes,
        )

    except VINDecodeError as e:
        logger.warning(
            "VIN decode error",
            extra={"error_message": sanitize_exception(e), "vin": sanitize_log(request.vin)},
        )
        raise VINValidationException(
            message=str(e),
            vin=request.vin,
        )

    except DiagnosisServiceError as e:
        logger.error(
            "Diagnosis service error", extra={"error_message": str(e), "details": e.details}
        )
        raise DiagnosisException(
            message="Hiba tortent a diagnosztika soran. Kerem, probalkozzon ujra.",
            details=e.details,
            original_error=e,
        )

    except AutoCognitixException:
        # Re-raise our custom exceptions to be handled by global handler
        raise

    except Exception as e:
        logger.exception(
            "Unexpected error in analyze_vehicle",
            extra={"error_type": type(e).__name__, "error_message": str(e)},
        )
        raise DiagnosisException(
            message="Varatlan hiba tortent a diagnosztika soran.",
            original_error=e,
        )


@router.get("/{diagnosis_id}", response_model=DiagnosisResponse)
async def get_diagnosis(
    diagnosis_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user_from_token),
):
    """
    Get a specific diagnosis by ID.

    Requires authentication - users can only access their own diagnoses.

    Args:
        diagnosis_id: UUID of the diagnosis
        db: Database session
        current_user: Authenticated user from JWT token

    Returns:
        The diagnosis result

    Raises:
        401: Not authenticated
        403: Forbidden - diagnosis belongs to another user
        404: Diagnosis not found
    """
    try:
        async with DiagnosisService(db) as service:
            # Ownership check happens in service layer (IDOR protection)
            result = await service.get_diagnosis_by_id(
                diagnosis_id,
                user_id=UUID(str(current_user.id)),
            )

            if result is None:
                # Generic message - don't reveal if diagnosis exists or not
                raise NotFoundException(
                    message="Diagnozis nem talalhato",
                    resource_type="diagnosis",
                    resource_id=str(diagnosis_id),
                )

            return result

    except AutoCognitixException:
        raise

    except HTTPException:
        raise

    except Exception as e:
        logger.exception(
            "Error retrieving diagnosis",
            extra={"diagnosis_id": sanitize_log(str(diagnosis_id)), "error_message": sanitize_exception(e)},
        )
        raise DiagnosisException(
            message="Hiba tortent a diagnozis lekeresekor.",
            details={"diagnosis_id": str(diagnosis_id)},
            original_error=e,
        )


@router.get(
    "/history/list",
    response_model=PaginatedDiagnosisHistory,
    summary="Get diagnosis history",
    description="""
**Get paginated diagnosis history** with optional filters.

Supports filtering by:
- `vehicle_make`: Filter by vehicle manufacturer (partial match)
- `vehicle_model`: Filter by vehicle model (partial match)
- `vehicle_year`: Filter by exact vehicle year
- `dtc_code`: Filter by DTC code (checks if any of the codes match)
- `date_from`: Filter by start date (ISO format)
- `date_to`: Filter by end date (ISO format)

**Pagination:**
- `skip`: Number of records to skip (default: 0)
- `limit`: Maximum records per page (1-100, default: 10)

**Requires authentication** - returns diagnoses for the authenticated user only.
    """,
)
async def get_diagnosis_history(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(10, ge=1, le=100, description="Maximum records to return"),
    vehicle_make: Optional[str] = Query(None, max_length=100, description="Filter by vehicle make"),
    vehicle_model: Optional[str] = Query(
        None, max_length=100, description="Filter by vehicle model"
    ),
    vehicle_year: Optional[int] = Query(
        None, ge=1900, le=2030, description="Filter by vehicle year"
    ),
    dtc_code: Optional[str] = Query(None, max_length=10, description="Filter by DTC code"),
    date_from: Optional[datetime] = Query(None, description="Filter by start date"),
    date_to: Optional[datetime] = Query(None, description="Filter by end date"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user_from_token),
):
    """
    Get diagnosis history with optional filters and pagination.

    Requires authentication - returns diagnosis history for the authenticated user.

    Args:
        skip: Number of records to skip (pagination)
        limit: Maximum number of records to return
        vehicle_make: Optional filter by vehicle make
        vehicle_model: Optional filter by vehicle model
        vehicle_year: Optional filter by vehicle year
        dtc_code: Optional filter by DTC code
        date_from: Optional filter by start date
        date_to: Optional filter by end date
        db: Database session
        current_user: Authenticated user from JWT token

    Returns:
        Paginated list of diagnoses with total count
    """
    try:
        repository = DiagnosisSessionRepository(db)
        user_id = UUID(str(current_user.id))

        items, total = await repository.get_filtered_history(
            user_id=user_id,
            vehicle_make=vehicle_make,
            vehicle_model=vehicle_model,
            vehicle_year=vehicle_year,
            dtc_code=dtc_code,
            date_from=date_from,
            date_to=date_to,
            skip=skip,
            limit=limit,
        )

        history_items = [
            DiagnosisHistoryItem(
                id=UUID(str(item.id)),
                vehicle_make=item.vehicle_make,
                vehicle_model=item.vehicle_model,
                vehicle_year=item.vehicle_year,
                vehicle_vin=item.vehicle_vin,
                dtc_codes=item.dtc_codes,
                symptoms_text=item.symptoms_text,
                confidence_score=item.confidence_score,
                created_at=item.created_at,
            )
            for item in items
        ]

        return PaginatedDiagnosisHistory(
            items=history_items,
            total=total,
            skip=skip,
            limit=limit,
            has_more=(skip + len(items)) < total,
        )

    except Exception as e:
        logger.exception("Error retrieving diagnosis history", extra={"error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving diagnosis history.",
        )


@router.delete(
    "/{diagnosis_id}",
    response_model=DeleteResponse,
    summary="Delete a diagnosis",
    description="""
**Soft delete a diagnosis** by ID.

The diagnosis is not permanently removed but marked as deleted.
This allows for potential recovery and audit purposes.

**Requires authentication** - users can only delete their own diagnoses.
    """,
)
async def delete_diagnosis(
    diagnosis_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user_from_token),
):
    """
    Soft delete a diagnosis by ID.

    Requires authentication - users can only delete their own diagnoses.

    Args:
        diagnosis_id: UUID of the diagnosis to delete
        db: Database session
        current_user: Authenticated user from JWT token

    Returns:
        Delete confirmation with the deleted ID

    Raises:
        404: Diagnosis not found or not owned by user
    """
    try:
        repository = DiagnosisSessionRepository(db)
        user_id = UUID(str(current_user.id))

        success = await repository.soft_delete(diagnosis_id, user_id=user_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Diagnosis {diagnosis_id} not found or already deleted",
            )

        await db.commit()

        # Invalidate cached diagnosis data
        try:
            from app.db.redis_cache import get_cache_service

            cache = await get_cache_service()
            if not cache.is_circuit_open():
                await cache.delete_pattern(f"api:diagnosis:{diagnosis_id}*")
        except Exception:
            pass  # Best-effort cache cleanup

        return DeleteResponse(
            success=True,
            message="Diagnosis successfully deleted",
            deleted_id=diagnosis_id,
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.exception(
            "Error deleting diagnosis",
            extra={"diagnosis_id": sanitize_log(str(diagnosis_id)), "error": sanitize_exception(e)},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while deleting the diagnosis.",
        )


@router.get(
    "/stats/summary",
    response_model=DiagnosisStats,
    summary="Get diagnosis statistics",
    description="""
**Get diagnosis statistics** for the current user.

Returns:
- Total number of diagnoses
- Average confidence score
- Most diagnosed vehicles (top 5)
- Most common DTC codes (top 10)
- Diagnosis counts by month (last 12 months)

**Requires authentication** - returns statistics for the authenticated user only.
    """,
)
async def get_diagnosis_stats(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user_from_token),
):
    """
    Get diagnosis statistics for the current user.

    Requires authentication.

    Args:
        db: Database session
        current_user: Authenticated user from JWT token

    Returns:
        Statistics including totals, averages, and trends
    """
    try:
        repository = DiagnosisSessionRepository(db)
        user_id = UUID(str(current_user.id))

        # Get user stats from repository
        stats = await repository.get_user_stats(user_id)

        # Get DTC frequency
        dtc_frequency = await repository.get_dtc_frequency(user_id=user_id, limit=10)

        return DiagnosisStats(
            total_diagnoses=stats["total_diagnoses"],
            avg_confidence=stats["avg_confidence"],
            most_diagnosed_vehicles=[
                VehicleDiagnosisCount(make=v["make"], model=v["model"], count=v["count"])
                for v in stats["most_diagnosed_vehicles"]
            ],
            most_common_dtcs=[
                DTCFrequency(code=d["code"], count=d["count"]) for d in dtc_frequency
            ],
            diagnoses_by_month=[
                MonthlyDiagnosisCount(
                    month=m["month"][:7] if m["month"] else "unknown",
                    count=m["count"],
                )
                for m in stats["diagnoses_by_month"]
            ],
        )

    except Exception as e:
        logger.exception("Error retrieving diagnosis stats", extra={"error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving diagnosis statistics.",
        )


@router.post(
    "/quick-analyze",
    responses=QUICK_ANALYZE_RESPONSES,
    summary="Quick DTC code lookup",
    description="""
**Quick analysis endpoint** - Fast DTC code lookup without full AI analysis.

This simplified endpoint provides basic DTC information without requiring
vehicle details. Ideal for quick lookups and previews.

**Query Parameters:**
- `dtc_codes`: Array of DTC codes (1-10 codes)

**Example:** `POST /api/v1/diagnosis/quick-analyze?dtc_codes=P0101&dtc_codes=P0171`

**Returns:**
- Basic DTC descriptions in Hungarian
- Severity level
- Common symptoms and causes

**Note:** For comprehensive AI-powered diagnosis with repair recommendations,
use the `/analyze` endpoint with full vehicle information.
    """,
)
async def quick_analyze(
    dtc_codes: List[str] = Query(..., min_length=1, max_length=10),
    db: AsyncSession = Depends(get_db),
):
    """
    Quick analysis endpoint for single DTC code lookup.

    This is a simplified endpoint that doesn't require vehicle info
    and returns basic DTC information without full RAG analysis.

    Args:
        dtc_codes: List of DTC codes to analyze
        db: Database session

    Returns:
        Basic DTC information and common causes
    """
    try:
        async with DiagnosisService(db) as service:
            # Validate DTC codes
            for code in dtc_codes:
                code = code.upper().strip()
                if not (len(code) == 5 and code[0] in "PBCU" and code[1:].isdigit()):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Invalid DTC code format: {code}",
                    )

            # Get DTC details from repository
            dtc_details = []
            for code in dtc_codes:
                details = await service.dtc_repository.get_by_code(code)
                if details:
                    dtc_details.append(
                        {
                            "code": details.code,
                            "description": details.description_hu or details.description_en,
                            "severity": details.severity,
                            "symptoms": details.symptoms,
                            "possible_causes": details.possible_causes,
                        }
                    )
                else:
                    dtc_details.append(
                        {
                            "code": code,
                            "description": "Hibakód nem található az adatbázisban",
                            "severity": "unknown",
                            "symptoms": [],
                            "possible_causes": [],
                        }
                    )

            return {
                "dtc_codes": dtc_details,
                "message": "Részletes diagnózishoz használja a /analyze végpontot járműadatokkal.",
            }

    except HTTPException:
        raise

    except Exception as e:
        logger.exception("Error in quick analyze", extra={"error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during quick analysis.",
        )


# =============================================================================
# Streaming Diagnosis Endpoint
# =============================================================================


async def _stream_results(
    rag_result: Dict[str, Any],
    diagnosis_id: UUID,
) -> Any:
    """Stream probable causes, repairs, and safety warnings as SSE events."""
    # Probable causes
    probable_causes = rag_result.get("probable_causes", [])
    for idx, cause in enumerate(probable_causes[:5]):
        yield _format_sse_event(
            StreamingEvent(
                event_type="cause",
                data={
                    "index": idx + 1,
                    "title": cause.get("title", ""),
                    "description": cause.get("description", ""),
                    "confidence": cause.get("confidence", 0.5),
                    "related_dtc_codes": cause.get("related_dtc_codes", []),
                    "components": cause.get("components", []),
                },
                diagnosis_id=diagnosis_id,
                progress=0.60 + (0.15 * (idx + 1) / max(len(probable_causes), 1)),
            )
        )
        await asyncio.sleep(0.02)

    # Repair recommendations
    repairs = rag_result.get("recommended_repairs", [])
    for idx, repair in enumerate(repairs[:5]):
        yield _format_sse_event(
            StreamingEvent(
                event_type="repair",
                data={
                    "index": idx + 1,
                    "title": repair.get("title", ""),
                    "description": repair.get("description", ""),
                    "difficulty": repair.get("difficulty", "intermediate"),
                    "estimated_cost_min": repair.get("estimated_cost_min"),
                    "estimated_cost_max": repair.get("estimated_cost_max"),
                    "estimated_time_minutes": repair.get("estimated_time_minutes"),
                },
                diagnosis_id=diagnosis_id,
                progress=0.75 + (0.17 * (idx + 1) / max(len(repairs), 1)),
            )
        )
        await asyncio.sleep(0.02)

    # Safety warnings
    safety_warnings = rag_result.get("safety_warnings", [])
    if safety_warnings:
        yield _format_sse_event(
            StreamingEvent(
                event_type="warning",
                data={"warnings": safety_warnings, "count": len(safety_warnings)},
                diagnosis_id=diagnosis_id,
                progress=0.95,
            )
        )


STREAMING_RESPONSES: Dict[Union[int, str], Dict[str, Any]] = {
    200: {
        "description": "Streaming diagnosis response (Server-Sent Events)",
        "content": {
            "text/event-stream": {
                "example": """event: start
data: {"event_type": "start", "diagnosis_id": "550e8400-e29b-41d4-a716-446655440000", "progress": 0.0}

event: context
data: {"event_type": "context", "data": {"dtc_count": 2, "symptom_matches": 5}, "progress": 0.2}

event: cause
data: {"event_type": "cause", "data": {"title": "MAF szenzor hiba", "confidence": 0.85}, "progress": 0.5}

event: complete
data: {"event_type": "complete", "data": {"confidence_score": 0.82}, "progress": 1.0}
"""
            }
        },
    },
    400: {
        "description": "Invalid request data",
    },
}


@router.post(
    "/analyze/stream",
    responses=STREAMING_RESPONSES,
    summary="Streaming vehicle analysis (SSE)",
    description="""
**Streaming diagnostic endpoint** - Real-time AI-powered diagnosis with Server-Sent Events.

This endpoint provides real-time streaming of the diagnosis process:

1. **start**: Diagnosis session started, returns diagnosis_id
2. **context**: Context retrieval progress (DTC matches, symptom matches, recalls)
3. **analysis**: AI analysis progress with partial results
4. **cause**: Each probable cause as it's identified
5. **repair**: Each repair recommendation as it's generated
6. **warning**: Safety warnings if any
7. **complete**: Final diagnosis complete with summary
8. **error**: Error occurred (if any)

**Event Format (Server-Sent Events):**
```
event: {event_type}
data: {json_data}

```

**Usage with JavaScript:**
```javascript
const eventSource = new EventSource('/api/v1/diagnosis/analyze/stream', {
    method: 'POST',
    body: JSON.stringify(requestData)
});
eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(data.event_type, data.data);
};
```

**Recommended for:** Long-running diagnoses where real-time feedback improves UX.
    """,
)
async def analyze_vehicle_stream(  # noqa: PLR0915
    request: DiagnosisStreamRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_current_user),
):
    """
    Streaming vehicle analysis endpoint.

    Returns Server-Sent Events (SSE) stream with real-time diagnosis progress.
    Protected by:
    - Per-stream timeout (STREAM_TIMEOUT_SECONDS)
    - Concurrent stream semaphore (MAX_CONCURRENT_STREAMS)
    - Client disconnection detection
    - Rate limiting via middleware (5 req/min)

    Args:
        request: Streaming diagnosis request with vehicle info, DTCs, and symptoms
        http_request: Raw HTTP request for disconnect detection
        db: Database session
        current_user: Optional authenticated user

    Returns:
        StreamingResponse with SSE events
    """
    diagnosis_id = uuid4()
    user_id = UUID(str(current_user.id)) if current_user else None

    # Capture request options for use in generator (avoid stale closure)
    include_context = request.include_context
    include_progress = request.include_progress

    async def _is_connected() -> bool:
        """Check if the client is still connected."""
        return not await http_request.is_disconnected()

    async def _run_pipeline(service):
        """Run the streaming diagnosis pipeline, yielding SSE events."""
        # Step 1: VIN decoding (if provided)
        vin_data = None
        if request.vin:
            try:
                vin_data = await service._decode_vin(request.vin)
                if include_context:
                    yield _format_sse_event(
                        StreamingEvent(
                            event_type="context",
                            data={
                                "stage": "vin_decode",
                                "message": f"VIN dekodolva: {vin_data.make} {vin_data.model}",
                            },
                            diagnosis_id=diagnosis_id,
                            progress=0.1,
                        )
                    )
            except Exception as e:
                logger.warning(
                    "VIN decode failed in stream",
                    extra={"diagnosis_id": str(diagnosis_id), "error": str(e)},
                )

        if not await _is_connected():
            return

        # Step 2: DTC validation
        dtc_details = await service._validate_and_enrich_dtc_codes(request.dtc_codes)
        if include_context:
            yield _format_sse_event(
                StreamingEvent(
                    event_type="context",
                    data={
                        "stage": "dtc_validation",
                        "message": f"{len(dtc_details)} DTC kod validalva",
                        "validated_codes": [d.code for d in dtc_details],
                    },
                    diagnosis_id=diagnosis_id,
                    progress=0.2,
                )
            )

        # Step 3: Symptom preprocessing
        preprocessed_symptoms = service._preprocess_symptoms(request.symptoms)
        if include_context:
            yield _format_sse_event(
                StreamingEvent(
                    event_type="context",
                    data={"stage": "symptom_preprocessing", "message": "Tunetek feldolgozva"},
                    diagnosis_id=diagnosis_id,
                    progress=0.25,
                )
            )

        if not await _is_connected():
            return

        # Step 4: NHTSA data fetch
        recalls, complaints = await service._fetch_nhtsa_data(
            make=request.vehicle_make,
            model=request.vehicle_model,
            year=request.vehicle_year,
        )
        if (recalls or complaints) and include_context:
            yield _format_sse_event(
                StreamingEvent(
                    event_type="context",
                    data={
                        "stage": "nhtsa_data",
                        "message": f"{len(recalls)} visszahivas, {len(complaints)} panasz talalva",
                        "recalls_found": len(recalls),
                        "complaints_found": len(complaints),
                    },
                    diagnosis_id=diagnosis_id,
                    progress=0.35,
                )
            )

        if not await _is_connected():
            return

        # Step 5: RAG pipeline
        if include_progress:
            yield _format_sse_event(
                StreamingEvent(
                    event_type="analysis",
                    data={"stage": "rag_start", "message": "AI elemzes inditasa..."},
                    diagnosis_id=diagnosis_id,
                    progress=0.4,
                )
            )

        from app.api.v1.schemas.diagnosis import DiagnosisRequest as DR

        diag_request = DR(
            vehicle_make=request.vehicle_make,
            vehicle_model=request.vehicle_model,
            vehicle_year=request.vehicle_year,
            vehicle_engine=request.vehicle_engine,
            vin=request.vin,
            dtc_codes=request.dtc_codes,
            symptoms=request.symptoms,
            additional_context=request.additional_context,
        )
        rag_result = await service._run_rag_pipeline(
            request=diag_request,
            dtc_details=dtc_details,
            preprocessed_symptoms=preprocessed_symptoms,
            recalls=recalls,
            complaints=complaints,
            vin_data=vin_data,
        )

        if not await _is_connected():
            return

        if include_progress:
            yield _format_sse_event(
                StreamingEvent(
                    event_type="analysis",
                    data={
                        "stage": "rag_complete",
                        "message": "AI elemzes kesz",
                        "model_used": rag_result.get("model_used", "unknown"),
                    },
                    diagnosis_id=diagnosis_id,
                    progress=0.6,
                )
            )

        # Step 6: Stream results (causes, repairs, warnings)
        async for event in _stream_results(rag_result, diagnosis_id):
            yield event

        # Step 7: Build, save, and complete
        response = service._build_response(
            diagnosis_id=diagnosis_id,
            request=diag_request,
            dtc_details=dtc_details,
            rag_result=rag_result,
            recalls=recalls,
            complaints=complaints,
        )
        await service._save_diagnosis_session(
            diagnosis_id=diagnosis_id,
            request=diag_request,
            response=response,
            user_id=user_id,
        )
        yield _format_sse_event(
            StreamingEvent(
                event_type="complete",
                data={
                    "diagnosis_id": str(diagnosis_id),
                    "confidence_score": response.confidence_score,
                    "urgency_level": response.urgency_level,
                    "probable_causes_count": len(response.probable_causes),
                    "repairs_count": len(response.recommended_repairs),
                    "recalls_count": len(response.related_recalls),
                    "complaints_count": len(response.similar_complaints),
                    "message": "Diagnosztika befejezve",
                },
                diagnosis_id=diagnosis_id,
                progress=1.0,
            )
        )

    async def generate_events():
        """Generate SSE events with semaphore and error handling."""
        acquired = False
        try:
            try:
                await asyncio.wait_for(_stream_semaphore.acquire(), timeout=10.0)
                acquired = True
            except asyncio.TimeoutError:
                yield _format_sse_event(
                    StreamingEvent(
                        event_type="error",
                        data={
                            "error_type": "capacity",
                            "message": "A szerver jelenleg tul van terhelve.",
                        },
                        diagnosis_id=diagnosis_id,
                        progress=0.0,
                    )
                )
                return

            yield _format_sse_event(
                StreamingEvent(
                    event_type="start",
                    data={
                        "message": "Diagnosztika inditasa...",
                        "vehicle": f"{request.vehicle_make} {request.vehicle_model} ({request.vehicle_year})",
                        "dtc_count": len(request.dtc_codes),
                    },
                    diagnosis_id=diagnosis_id,
                    progress=0.0,
                )
            )
            await asyncio.sleep(0.1)

            async with DiagnosisService(db) as service:
                if not await _is_connected():
                    return
                async for event in _run_pipeline(service):
                    yield event

        except DTCValidationError as e:
            yield _format_sse_event(
                StreamingEvent(
                    event_type="error",
                    data={"error_type": "dtc_validation", "message": str(e)},
                    diagnosis_id=diagnosis_id,
                    progress=0.0,
                )
            )
        except VINDecodeError:
            yield _format_sse_event(
                StreamingEvent(
                    event_type="error",
                    data={
                        "error_type": "vin_decode",
                        "message": "VIN dekodolasa sikertelen. Kerem, ellenorizze a VIN szamot.",
                    },
                    diagnosis_id=diagnosis_id,
                    progress=0.0,
                )
            )
        except asyncio.CancelledError:
            logger.info(f"Stream cancelled for diagnosis {diagnosis_id}")
            return
        except Exception as e:
            logger.exception(
                "Streaming diagnosis error",
                extra={"diagnosis_id": str(diagnosis_id), "error_type": type(e).__name__},
            )
            try:
                yield _format_sse_event(
                    StreamingEvent(
                        event_type="error",
                        data={
                            "error_type": "internal",
                            "message": "Varatlan hiba tortent a diagnosztika soran.",
                        },
                        diagnosis_id=diagnosis_id,
                        progress=0.0,
                    )
                )
            except Exception:
                logger.error(f"Failed to send error event for stream {diagnosis_id}")
        finally:
            if acquired:
                _stream_semaphore.release()

    async def _timeout_wrapper():
        """Enforce stream timeout to prevent Slowloris attacks."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + STREAM_TIMEOUT_SECONDS
        gen = generate_events()
        try:
            async for event in gen:
                if loop.time() > deadline:
                    logger.warning(f"Stream timeout reached for diagnosis {diagnosis_id}")
                    yield _format_sse_event(
                        StreamingEvent(
                            event_type="error",
                            data={
                                "error_type": "timeout",
                                "message": "A diagnosztika tullepte a maximalis idokeretet.",
                            },
                            diagnosis_id=diagnosis_id,
                            progress=0.0,
                        )
                    )
                    # Cancel the generator to release resources (semaphore, DB session)
                    await gen.aclose()
                    return
                yield event
        except GeneratorExit:
            await gen.aclose()

    return StreamingResponse(
        _timeout_wrapper(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _format_sse_event(event: StreamingEvent) -> str:
    """Format a StreamingEvent as SSE format."""
    data = {
        "event_type": event.event_type,
        "data": event.data,
        "diagnosis_id": str(event.diagnosis_id),
        "timestamp": event.timestamp.isoformat().replace("+00:00", "Z"),
        "progress": event.progress,
    }
    return f"event: {event.event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
