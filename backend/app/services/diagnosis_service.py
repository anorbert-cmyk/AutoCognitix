"""
Diagnosis Service - Core diagnostic orchestration service.

This module provides the main diagnosis service that orchestrates:
- VIN decoding via NHTSA API
- DTC code validation and lookup
- Hungarian NLP preprocessing
- NHTSA recalls and complaints checking
- RAG pipeline for intelligent diagnosis
- Result persistence to database

Author: AutoCognitix Team
"""

import asyncio
from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.schemas.diagnosis import (
    DiagnosisHistoryItem,
    DiagnosisRequest,
    DiagnosisResponse,
    ProbableCause,
    RepairRecommendation,
    Source,
)
from app.core.logging import get_logger
from app.db.postgres.models import DTCCode
from app.db.postgres.repositories import DiagnosisSessionRepository, DTCCodeRepository
from app.services.embedding_service import preprocess_hungarian
from app.services.nhtsa_service import (
    Complaint,
    NHTSAError,
    NHTSAService,
    Recall,
    VINDecodeResult,
    get_nhtsa_service,
)

logger = get_logger(__name__)


class DiagnosisServiceError(Exception):
    """Custom exception for diagnosis service errors."""

    def __init__(self, message: str, details: dict | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class DTCValidationError(DiagnosisServiceError):
    """Exception raised when DTC code validation fails."""

    pass


class VINDecodeError(DiagnosisServiceError):
    """Exception raised when VIN decoding fails."""

    pass


class DiagnosisService:
    """
    Main diagnosis orchestration service.

    Coordinates all components to provide comprehensive vehicle diagnostics:
    - VIN decoding for vehicle identification
    - DTC code validation and enrichment
    - Hungarian symptom preprocessing
    - NHTSA safety data integration
    - RAG-based intelligent diagnosis
    - Result persistence

    Usage:
        async with DiagnosisService(db) as service:
            result = await service.analyze_vehicle(request)
    """

    def __init__(self, db: AsyncSession):
        """
        Initialize the diagnosis service.

        Args:
            db: SQLAlchemy async session for database operations.
        """
        self.db = db
        self._nhtsa_service: NHTSAService | None = None
        self._dtc_repository: DTCCodeRepository | None = None
        self._diagnosis_repository: DiagnosisSessionRepository | None = None

    async def _get_nhtsa_service(self) -> NHTSAService:
        """Get or create NHTSA service instance."""
        if self._nhtsa_service is None:
            self._nhtsa_service = await get_nhtsa_service()
        return self._nhtsa_service

    @property
    def dtc_repository(self) -> DTCCodeRepository:
        """Get DTC code repository."""
        if self._dtc_repository is None:
            self._dtc_repository = DTCCodeRepository(self.db)
        return self._dtc_repository

    @property
    def diagnosis_repository(self) -> DiagnosisSessionRepository:
        """Get diagnosis session repository."""
        if self._diagnosis_repository is None:
            self._diagnosis_repository = DiagnosisSessionRepository(self.db)
        return self._diagnosis_repository

    async def __aenter__(self) -> "DiagnosisService":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        # NHTSA service cleanup is handled at the application level
        pass

    # =========================================================================
    # Main Analysis Method
    # =========================================================================

    async def analyze_vehicle(
        self,
        request: DiagnosisRequest,
        user_id: UUID | None = None,
    ) -> DiagnosisResponse:
        """
        Perform comprehensive vehicle diagnosis.

        This method orchestrates the complete diagnosis workflow:
        1. VIN decoding (if VIN provided)
        2. DTC code validation and enrichment
        3. Hungarian symptom preprocessing
        4. NHTSA recalls and complaints check
        5. RAG pipeline execution
        6. Result persistence
        7. Response assembly

        Args:
            request: DiagnosisRequest with vehicle info, DTC codes, and symptoms.
            user_id: Optional user ID for session tracking.

        Returns:
            DiagnosisResponse with probable causes, recommendations, and sources.

        Raises:
            DiagnosisServiceError: If diagnosis fails.
            DTCValidationError: If DTC codes are invalid.
            VINDecodeError: If VIN decoding fails.
        """
        diagnosis_id = uuid4()
        logger.info(
            f"Starting diagnosis {diagnosis_id} for {request.vehicle_make} "
            f"{request.vehicle_model} {request.vehicle_year}"
        )

        try:
            # Step 1: VIN decoding (if provided)
            vin_data: VINDecodeResult | None = None
            if request.vin:
                vin_data = await self._decode_vin(request.vin)
                logger.info(f"VIN decoded: {vin_data.make} {vin_data.model}")

            # Step 2: Validate and enrich DTC codes
            dtc_details = await self._validate_and_enrich_dtc_codes(request.dtc_codes)
            logger.info(f"Validated {len(dtc_details)} DTC codes")

            # Step 3: Preprocess Hungarian symptoms
            preprocessed_symptoms = self._preprocess_symptoms(request.symptoms)
            logger.debug(f"Preprocessed symptoms: {preprocessed_symptoms[:100]}...")

            # Step 4: Fetch NHTSA recalls and complaints (parallel)
            recalls, complaints = await self._fetch_nhtsa_data(
                make=request.vehicle_make,
                model=request.vehicle_model,
                year=request.vehicle_year,
            )
            logger.info(
                f"Found {len(recalls)} recalls and {len(complaints)} complaints"
            )

            # Step 5: Run RAG pipeline
            rag_result = await self._run_rag_pipeline(
                request=request,
                dtc_details=dtc_details,
                preprocessed_symptoms=preprocessed_symptoms,
                recalls=recalls,
                complaints=complaints,
                vin_data=vin_data,
            )

            # Step 6: Build response
            response = self._build_response(
                diagnosis_id=diagnosis_id,
                request=request,
                dtc_details=dtc_details,
                rag_result=rag_result,
                recalls=recalls,
                complaints=complaints,
            )

            # Step 7: Save to database
            await self._save_diagnosis_session(
                diagnosis_id=diagnosis_id,
                request=request,
                response=response,
                user_id=user_id,
            )

            logger.info(
                f"Diagnosis {diagnosis_id} completed with confidence "
                f"{response.confidence_score:.2f}"
            )
            return response

        except DiagnosisServiceError:
            raise
        except Exception as e:
            logger.error(f"Diagnosis {diagnosis_id} failed: {e}", exc_info=True)
            raise DiagnosisServiceError(
                message=f"Diagnosis failed: {e!s}",
                details={"diagnosis_id": str(diagnosis_id)},
            )

    # =========================================================================
    # VIN Decoding
    # =========================================================================

    async def _decode_vin(self, vin: str) -> VINDecodeResult:
        """
        Decode VIN using NHTSA API.

        Args:
            vin: 17-character VIN string.

        Returns:
            VINDecodeResult with vehicle information.

        Raises:
            VINDecodeError: If VIN decoding fails.
        """
        try:
            nhtsa = await self._get_nhtsa_service()
            result = await nhtsa.decode_vin(vin)

            if not result.is_valid:
                raise VINDecodeError(
                    message=f"Invalid VIN: {result.error_text}",
                    details={"vin": vin, "error_code": result.error_code},
                )

            return result

        except NHTSAError as e:
            logger.error(f"NHTSA VIN decode error: {e}")
            raise VINDecodeError(
                message=f"VIN decoding service error: {e.message}",
                details={"vin": vin},
            )
        except ValueError as e:
            raise VINDecodeError(
                message=str(e),
                details={"vin": vin},
            )

    # =========================================================================
    # DTC Code Validation
    # =========================================================================

    async def _validate_and_enrich_dtc_codes(
        self, dtc_codes: list[str]
    ) -> list[DTCCode]:
        """
        Validate DTC codes and fetch their details from database.

        Args:
            dtc_codes: List of DTC code strings to validate.

        Returns:
            List of DTCCode model instances with full details.

        Note:
            Unknown DTC codes are logged but not rejected,
            allowing diagnosis to proceed with partial information.
        """
        validated_codes: list[DTCCode] = []
        unknown_codes: list[str] = []

        for code in dtc_codes:
            code_upper = code.upper().strip()

            # Basic format validation
            if not self._is_valid_dtc_format(code_upper):
                logger.warning(f"Invalid DTC format: {code}")
                continue

            # Fetch from database
            dtc_detail = await self.dtc_repository.get_by_code(code_upper)

            if dtc_detail:
                validated_codes.append(dtc_detail)
            else:
                unknown_codes.append(code_upper)
                logger.info(f"Unknown DTC code (not in database): {code_upper}")

        if unknown_codes:
            logger.warning(
                f"Proceeding with {len(unknown_codes)} unknown DTC codes: "
                f"{', '.join(unknown_codes)}"
            )

        return validated_codes

    @staticmethod
    def _is_valid_dtc_format(code: str) -> bool:
        """
        Validate DTC code format.

        Standard DTC format: Letter + 4 digits (e.g., P0101, B1234, C0456, U1000)
        - P: Powertrain
        - B: Body
        - C: Chassis
        - U: Network/Communication

        Args:
            code: DTC code string to validate.

        Returns:
            True if format is valid, False otherwise.
        """
        if len(code) != 5:
            return False

        if code[0] not in "PBCU":
            return False

        return code[1:].isdigit()

    # =========================================================================
    # Symptom Preprocessing
    # =========================================================================

    def _preprocess_symptoms(self, symptoms: str) -> str:
        """
        Preprocess Hungarian symptom text.

        Applies Hungarian NLP preprocessing including:
        - Lemmatization
        - Stopword removal
        - Punctuation removal

        Args:
            symptoms: Raw Hungarian symptom description.

        Returns:
            Preprocessed symptom text.
        """
        try:
            return preprocess_hungarian(symptoms)
        except Exception as e:
            logger.warning(f"Symptom preprocessing failed: {e}, using raw text")
            return symptoms

    # =========================================================================
    # NHTSA Data Fetching
    # =========================================================================

    async def _fetch_nhtsa_data(
        self,
        make: str,
        model: str,
        year: int,
    ) -> tuple[list[Recall], list[Complaint]]:
        """
        Fetch recalls and complaints from NHTSA API in parallel.

        Args:
            make: Vehicle make.
            model: Vehicle model.
            year: Vehicle year.

        Returns:
            Tuple of (recalls list, complaints list).
        """
        nhtsa = await self._get_nhtsa_service()

        try:
            # Fetch recalls and complaints in parallel
            recalls_task = nhtsa.get_recalls(make, model, year)
            complaints_task = nhtsa.get_complaints(make, model, year)

            recalls, complaints = await asyncio.gather(
                recalls_task,
                complaints_task,
                return_exceptions=True,
            )

            # Handle exceptions gracefully
            if isinstance(recalls, Exception):
                logger.warning(f"Failed to fetch recalls: {recalls}")
                recalls = []

            if isinstance(complaints, Exception):
                logger.warning(f"Failed to fetch complaints: {complaints}")
                complaints = []

            return recalls, complaints

        except Exception as e:
            logger.error(f"NHTSA data fetch error: {e}")
            return [], []

    # =========================================================================
    # RAG Pipeline
    # =========================================================================

    async def _run_rag_pipeline(
        self,
        request: DiagnosisRequest,
        dtc_details: list[DTCCode],
        preprocessed_symptoms: str,
        recalls: list[Recall],
        complaints: list[Complaint],
        vin_data: VINDecodeResult | None,
    ) -> dict:
        """
        Execute the RAG (Retrieval-Augmented Generation) pipeline.

        This method integrates with the RAG service to generate
        intelligent diagnosis based on:
        - DTC code details
        - Preprocessed symptoms
        - NHTSA safety data
        - Vehicle-specific knowledge base

        Args:
            request: Original diagnosis request.
            dtc_details: Validated DTC code details.
            preprocessed_symptoms: Preprocessed Hungarian symptoms.
            recalls: NHTSA recalls for the vehicle.
            complaints: NHTSA complaints for the vehicle.
            vin_data: Optional VIN decode result.

        Returns:
            Dictionary with RAG pipeline results including:
            - probable_causes: List of probable cause dictionaries
            - recommended_repairs: List of repair recommendation dictionaries
            - confidence_score: Overall confidence (0-1)
            - sources: List of source dictionaries
        """
        try:
            # Lazy import to avoid circular dependency
            # and allow rag_service to be created by another agent
            from app.services.rag_service import diagnose

            # Build context for RAG
            context = self._build_rag_context(
                request=request,
                dtc_details=dtc_details,
                preprocessed_symptoms=preprocessed_symptoms,
                recalls=recalls,
                complaints=complaints,
                vin_data=vin_data,
            )

            # Execute RAG diagnosis
            result = await diagnose(context)
            return result

        except ImportError:
            logger.warning(
                "RAG service not available, using fallback diagnosis"
            )
            return self._fallback_diagnosis(
                dtc_details=dtc_details,
                recalls=recalls,
                complaints=complaints,
            )
        except Exception as e:
            logger.error(f"RAG pipeline error: {e}", exc_info=True)
            return self._fallback_diagnosis(
                dtc_details=dtc_details,
                recalls=recalls,
                complaints=complaints,
            )

    def _build_rag_context(
        self,
        request: DiagnosisRequest,
        dtc_details: list[DTCCode],
        preprocessed_symptoms: str,
        recalls: list[Recall],
        complaints: list[Complaint],
        vin_data: VINDecodeResult | None,
    ) -> dict:
        """
        Build context dictionary for RAG pipeline.

        Args:
            request: Original diagnosis request.
            dtc_details: Validated DTC codes.
            preprocessed_symptoms: Preprocessed symptoms.
            recalls: NHTSA recalls.
            complaints: NHTSA complaints.
            vin_data: Optional VIN data.

        Returns:
            Context dictionary for RAG service.
        """
        return {
            "vehicle": {
                "make": request.vehicle_make,
                "model": request.vehicle_model,
                "year": request.vehicle_year,
                "engine": request.vehicle_engine,
                "vin": request.vin,
                "vin_data": vin_data.model_dump() if vin_data else None,
            },
            "dtc_codes": [
                {
                    "code": dtc.code,
                    "description_en": dtc.description_en,
                    "description_hu": dtc.description_hu,
                    "category": dtc.category,
                    "severity": dtc.severity,
                    "symptoms": dtc.symptoms,
                    "possible_causes": dtc.possible_causes,
                    "diagnostic_steps": dtc.diagnostic_steps,
                }
                for dtc in dtc_details
            ],
            "symptoms": {
                "raw": request.symptoms,
                "preprocessed": preprocessed_symptoms,
            },
            "additional_context": request.additional_context,
            "nhtsa": {
                "recalls": [
                    {
                        "campaign_number": r.campaign_number,
                        "component": r.component,
                        "summary": r.summary,
                        "consequence": r.consequence,
                        "remedy": r.remedy,
                    }
                    for r in recalls
                ],
                "complaints": [
                    {
                        "components": c.components,
                        "summary": c.summary,
                        "crash": c.crash,
                        "fire": c.fire,
                    }
                    for c in complaints
                ],
            },
        }

    def _fallback_diagnosis(
        self,
        dtc_details: list[DTCCode],
        recalls: list[Recall],
        complaints: list[Complaint],
    ) -> dict:
        """
        Generate fallback diagnosis when RAG service is unavailable.

        Uses DTC code information and NHTSA data directly.

        Args:
            dtc_details: Validated DTC codes.
            recalls: NHTSA recalls.
            complaints: NHTSA complaints.

        Returns:
            Fallback diagnosis result dictionary.
        """
        probable_causes = []
        recommended_repairs = []
        sources = []

        # Generate causes from DTC codes
        for dtc in dtc_details:
            for idx, cause in enumerate(dtc.possible_causes[:3]):
                probable_causes.append({
                    "title": f"{dtc.code} - {cause[:50]}..." if len(cause) > 50 else f"{dtc.code} - {cause}",
                    "description": cause,
                    "confidence": max(0.3, 0.7 - (idx * 0.1)),
                    "related_dtc_codes": [dtc.code],
                    "components": [],
                })

            # Add diagnostic steps as repair recommendations
            for step in dtc.diagnostic_steps[:2]:
                recommended_repairs.append({
                    "title": f"Diagnozis: {dtc.code}",
                    "description": step,
                    "difficulty": "intermediate",
                    "parts_needed": [],
                })

            sources.append({
                "type": "database",
                "title": f"DTC Database - {dtc.code}",
                "url": None,
                "relevance_score": 0.8,
            })

        # Add recall information
        for recall in recalls[:3]:
            probable_causes.append({
                "title": f"NHTSA Recall: {recall.component}",
                "description": recall.summary,
                "confidence": 0.9,
                "related_dtc_codes": [],
                "components": [recall.component],
            })

            if recall.remedy:
                recommended_repairs.append({
                    "title": f"Recall javitas: {recall.component}",
                    "description": recall.remedy,
                    "difficulty": "professional",
                    "parts_needed": [],
                })

            sources.append({
                "type": "recall",
                "title": f"NHTSA Recall {recall.campaign_number}",
                "url": None,
                "relevance_score": 0.95,
            })

        # Calculate confidence based on available data
        confidence = 0.3
        if dtc_details:
            confidence += 0.2
        if recalls:
            confidence += 0.2
        if complaints:
            confidence += 0.1

        return {
            "probable_causes": probable_causes,
            "recommended_repairs": recommended_repairs,
            "confidence_score": min(0.8, confidence),
            "sources": sources,
        }

    # =========================================================================
    # Response Building
    # =========================================================================

    def _build_response(
        self,
        diagnosis_id: UUID,
        request: DiagnosisRequest,
        dtc_details: list[DTCCode],
        rag_result: dict,
        recalls: list[Recall],
        complaints: list[Complaint],
    ) -> DiagnosisResponse:
        """
        Build the final DiagnosisResponse object.

        Args:
            diagnosis_id: Unique diagnosis ID.
            request: Original request.
            dtc_details: Validated DTC codes.
            rag_result: RAG pipeline results.
            recalls: NHTSA recalls.
            complaints: NHTSA complaints.

        Returns:
            Complete DiagnosisResponse object.
        """
        # Build probable causes
        probable_causes = [
            ProbableCause(
                title=cause.get("title", "Unknown"),
                description=cause.get("description", ""),
                confidence=cause.get("confidence", 0.5),
                related_dtc_codes=cause.get("related_dtc_codes", []),
                components=cause.get("components", []),
            )
            for cause in rag_result.get("probable_causes", [])
        ]

        # Build repair recommendations
        recommended_repairs = [
            RepairRecommendation(
                title=repair.get("title", "Unknown"),
                description=repair.get("description", ""),
                estimated_cost_min=repair.get("estimated_cost_min"),
                estimated_cost_max=repair.get("estimated_cost_max"),
                estimated_cost_currency=repair.get("estimated_cost_currency", "HUF"),
                difficulty=repair.get("difficulty", "intermediate"),
                parts_needed=repair.get("parts_needed", []),
                estimated_time_minutes=repair.get("estimated_time_minutes"),
            )
            for repair in rag_result.get("recommended_repairs", [])
        ]

        # Build sources
        sources = [
            Source(
                type=source.get("type", "database"),
                title=source.get("title", "Unknown"),
                url=source.get("url"),
                relevance_score=source.get("relevance_score", 0.5),
            )
            for source in rag_result.get("sources", [])
        ]

        return DiagnosisResponse(
            id=diagnosis_id,
            vehicle_make=request.vehicle_make,
            vehicle_model=request.vehicle_model,
            vehicle_year=request.vehicle_year,
            dtc_codes=request.dtc_codes,
            symptoms=request.symptoms,
            probable_causes=probable_causes,
            recommended_repairs=recommended_repairs,
            confidence_score=rag_result.get("confidence_score", 0.5),
            sources=sources,
            created_at=datetime.utcnow(),
        )

    # =========================================================================
    # Database Operations
    # =========================================================================

    async def _save_diagnosis_session(
        self,
        diagnosis_id: UUID,
        request: DiagnosisRequest,
        response: DiagnosisResponse,
        user_id: UUID | None,
    ) -> None:
        """
        Save diagnosis session to database.

        Args:
            diagnosis_id: Unique diagnosis ID.
            request: Original request.
            response: Generated response.
            user_id: Optional user ID.
        """
        try:
            session_data = {
                "id": diagnosis_id,
                "user_id": user_id,
                "vehicle_make": request.vehicle_make,
                "vehicle_model": request.vehicle_model,
                "vehicle_year": request.vehicle_year,
                "vehicle_vin": request.vin,
                "dtc_codes": request.dtc_codes,
                "symptoms_text": request.symptoms,
                "additional_context": request.additional_context,
                "diagnosis_result": response.model_dump(mode="json"),
                "confidence_score": response.confidence_score,
            }

            await self.diagnosis_repository.create(session_data)
            logger.debug(f"Saved diagnosis session {diagnosis_id}")

        except Exception as e:
            logger.error(f"Failed to save diagnosis session: {e}", exc_info=True)
            # Don't raise - diagnosis should still be returned even if save fails

    async def get_diagnosis_by_id(self, diagnosis_id: UUID) -> DiagnosisResponse | None:
        """
        Retrieve a diagnosis by its ID.

        Args:
            diagnosis_id: UUID of the diagnosis to retrieve.

        Returns:
            DiagnosisResponse if found, None otherwise.
        """
        try:
            session = await self.diagnosis_repository.get(diagnosis_id)

            if not session:
                logger.debug(f"Diagnosis {diagnosis_id} not found")
                return None

            # Reconstruct response from stored data
            result = session.diagnosis_result

            return DiagnosisResponse(
                id=session.id,
                vehicle_make=session.vehicle_make,
                vehicle_model=session.vehicle_model,
                vehicle_year=session.vehicle_year,
                dtc_codes=session.dtc_codes,
                symptoms=session.symptoms_text,
                probable_causes=[
                    ProbableCause(**cause)
                    for cause in result.get("probable_causes", [])
                ],
                recommended_repairs=[
                    RepairRecommendation(**repair)
                    for repair in result.get("recommended_repairs", [])
                ],
                confidence_score=session.confidence_score,
                sources=[
                    Source(**source)
                    for source in result.get("sources", [])
                ],
                created_at=session.created_at,
            )

        except Exception as e:
            logger.error(f"Error retrieving diagnosis {diagnosis_id}: {e}")
            raise DiagnosisServiceError(
                message=f"Failed to retrieve diagnosis: {e!s}",
                details={"diagnosis_id": str(diagnosis_id)},
            )

    async def get_user_history(
        self,
        user_id: UUID,
        skip: int = 0,
        limit: int = 10,
    ) -> list[DiagnosisHistoryItem]:
        """
        Get diagnosis history for a user.

        Args:
            user_id: UUID of the user.
            skip: Number of records to skip (pagination).
            limit: Maximum number of records to return.

        Returns:
            List of DiagnosisHistoryItem objects.
        """
        try:
            sessions = await self.diagnosis_repository.get_user_history(
                user_id=user_id,
                skip=skip,
                limit=limit,
            )

            return [
                DiagnosisHistoryItem(
                    id=session.id,
                    vehicle_make=session.vehicle_make,
                    vehicle_model=session.vehicle_model,
                    vehicle_year=session.vehicle_year,
                    dtc_codes=session.dtc_codes,
                    confidence_score=session.confidence_score,
                    created_at=session.created_at,
                )
                for session in sessions
            ]

        except Exception as e:
            logger.error(f"Error retrieving user history for {user_id}: {e}")
            raise DiagnosisServiceError(
                message=f"Failed to retrieve diagnosis history: {e!s}",
                details={"user_id": str(user_id)},
            )


# =============================================================================
# Service Factory Functions
# =============================================================================


async def get_diagnosis_service(db: AsyncSession) -> DiagnosisService:
    """
    Get a DiagnosisService instance.

    This is a factory function for use with FastAPI's dependency injection.

    Args:
        db: SQLAlchemy async session.

    Returns:
        DiagnosisService instance.

    Usage:
        @router.post("/diagnose")
        async def diagnose(
            request: DiagnosisRequest,
            service: DiagnosisService = Depends(get_diagnosis_service),
        ):
            return await service.analyze_vehicle(request)
    """
    return DiagnosisService(db)


# =============================================================================
# Convenience Functions
# =============================================================================


async def analyze_vehicle(
    db: AsyncSession,
    request: DiagnosisRequest,
    user_id: UUID | None = None,
) -> DiagnosisResponse:
    """
    Convenience function for vehicle analysis.

    Args:
        db: SQLAlchemy async session.
        request: DiagnosisRequest object.
        user_id: Optional user ID.

    Returns:
        DiagnosisResponse object.
    """
    async with DiagnosisService(db) as service:
        return await service.analyze_vehicle(request, user_id)


async def get_diagnosis_by_id(
    db: AsyncSession,
    diagnosis_id: UUID,
) -> DiagnosisResponse | None:
    """
    Convenience function to get diagnosis by ID.

    Args:
        db: SQLAlchemy async session.
        diagnosis_id: UUID of the diagnosis.

    Returns:
        DiagnosisResponse if found, None otherwise.
    """
    async with DiagnosisService(db) as service:
        return await service.get_diagnosis_by_id(diagnosis_id)


async def get_user_history(
    db: AsyncSession,
    user_id: UUID,
    skip: int = 0,
    limit: int = 10,
) -> list[DiagnosisHistoryItem]:
    """
    Convenience function to get user diagnosis history.

    Args:
        db: SQLAlchemy async session.
        user_id: UUID of the user.
        skip: Pagination offset.
        limit: Maximum results.

    Returns:
        List of DiagnosisHistoryItem objects.
    """
    async with DiagnosisService(db) as service:
        return await service.get_user_history(user_id, skip, limit)
