"""
RAG (Retrieval-Augmented Generation) Service for AutoCognitix.

This module provides a complete Hungarian vehicle diagnostic RAG pipeline:
- Multi-source retrieval (Qdrant vectors, Neo4j graph, PostgreSQL text search)
- Hybrid ranking combining semantic and keyword search
- LLM provider abstraction (Anthropic, OpenAI, Ollama, rule-based fallback)
- Hungarian language prompt templates
- Response parsing and confidence scoring
- Async processing with caching

Author: AutoCognitix Team
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Optional

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.neo4j_models import get_diagnostic_path
from app.db.postgres.models import DTCCode, KnownIssue
from app.db.qdrant_client import QdrantService, qdrant_client
from app.prompts.diagnosis_hu import (
    SYSTEM_PROMPT_HU,
    DiagnosisPromptContext,
    ParsedDiagnosisResponse,
    build_diagnosis_prompt,
    format_dtc_context,
    format_recall_context,
    format_repair_context,
    format_symptom_context,
    generate_rule_based_diagnosis,
    parse_diagnosis_response,
)
from app.services.embedding_service import (
    embed_text,
    get_embedding_service,
    preprocess_hungarian,
)
from app.services.llm_provider import (
    LLMConfig,
    LLMProviderType,
    get_llm_provider,
    is_llm_available,
)

logger = get_logger(__name__)


# =============================================================================
# Enums and Data Classes
# =============================================================================


class ConfidenceLevel(StrEnum):
    """Confidence levels for diagnosis."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class RetrievalSource(StrEnum):
    """Sources for context retrieval."""
    QDRANT_DTC = "qdrant_dtc"
    QDRANT_SYMPTOM = "qdrant_symptom"
    NEO4J_GRAPH = "neo4j_graph"
    POSTGRES_TEXT = "postgres_text"
    NHTSA = "nhtsa"


@dataclass
class VehicleInfo:
    """Vehicle information for diagnosis."""
    make: str
    model: str
    year: int
    vin: str | None = None
    engine_code: str | None = None
    mileage_km: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "make": self.make,
            "model": self.model,
            "year": self.year,
            "vin": self.vin,
            "engine_code": self.engine_code,
            "mileage_km": self.mileage_km,
        }


@dataclass
class RetrievedItem:
    """Item retrieved from any source."""
    content: dict[str, Any]
    source: RetrievalSource
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGContext:
    """Context assembled for RAG generation."""
    dtc_items: list[RetrievedItem] = field(default_factory=list)
    symptom_items: list[RetrievedItem] = field(default_factory=list)
    graph_items: list[RetrievedItem] = field(default_factory=list)
    text_items: list[RetrievedItem] = field(default_factory=list)
    recall_items: list[RetrievedItem] = field(default_factory=list)

    # Formatted context strings
    dtc_context: str = ""
    symptom_context: str = ""
    repair_context: str = ""
    recall_context: str = ""

    # Graph data for structured access
    graph_data: dict[str, Any] = field(default_factory=dict)

    def get_all_items(self) -> list[RetrievedItem]:
        """Get all retrieved items from all sources."""
        return (
            self.dtc_items +
            self.symptom_items +
            self.graph_items +
            self.text_items +
            self.recall_items
        )

    def to_formatted_string(self) -> str:
        """Format all context for LLM prompt."""
        sections = []

        if self.dtc_context:
            sections.append(f"## DTC Kod Informaciok:\n{self.dtc_context}")

        if self.symptom_context:
            sections.append(f"## Hasonlo Tunetek:\n{self.symptom_context}")

        if self.repair_context:
            sections.append(f"## Kapcsolodo Javitasok:\n{self.repair_context}")

        if self.recall_context:
            sections.append(f"## Visszahivasok es Panaszok:\n{self.recall_context}")

        return "\n\n".join(sections) if sections else "Nincs elerheto kontextus."


@dataclass
class RepairRecommendation:
    """Repair recommendation with cost and time estimates."""
    name: str
    description: str
    difficulty: str
    estimated_time_minutes: int | None = None
    estimated_cost_min: int | None = None
    estimated_cost_max: int | None = None
    parts: list[dict[str, Any]] = field(default_factory=list)
    diagnostic_steps: list[str] = field(default_factory=list)


@dataclass
class DiagnosisResult:
    """Complete diagnosis result with confidence and recommendations."""
    dtc_codes: list[str]
    symptoms: str
    vehicle_info: VehicleInfo

    # Main diagnosis
    diagnosis_summary: str
    root_cause_analysis: str
    confidence: ConfidenceLevel
    confidence_score: float  # 0.0 - 1.0

    # Structured results
    probable_causes: list[dict[str, Any]] = field(default_factory=list)
    repair_recommendations: list[RepairRecommendation] = field(default_factory=list)
    safety_warnings: list[str] = field(default_factory=list)
    diagnostic_steps: list[str] = field(default_factory=list)

    # Context used
    similar_cases: list[dict[str, Any]] = field(default_factory=list)
    related_dtc_info: list[dict[str, Any]] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)

    # Metadata
    generated_at: datetime = field(default_factory=datetime.utcnow)
    model_used: str = ""
    provider_used: str = ""
    processing_time_ms: int = 0
    used_fallback: bool = False


# =============================================================================
# Hybrid Ranking
# =============================================================================


class HybridRanker:
    """
    Hybrid ranking combining multiple retrieval sources.

    Implements Reciprocal Rank Fusion (RRF) for combining
    semantic similarity scores with keyword match scores.
    """

    def __init__(self, k: int = 60):
        """
        Initialize the hybrid ranker.

        Args:
            k: RRF parameter (higher = more weight to lower ranks)
        """
        self.k = k

    def reciprocal_rank_fusion(
        self,
        ranked_lists: list[list[RetrievedItem]],
        weights: list[float] | None = None,
    ) -> list[RetrievedItem]:
        """
        Combine multiple ranked lists using RRF.

        Args:
            ranked_lists: List of ranked item lists from different sources.
            weights: Optional weights for each list (default: equal weights).

        Returns:
            Combined and re-ranked list of items.
        """
        if not ranked_lists:
            return []

        if weights is None:
            weights = [1.0] * len(ranked_lists)

        # Calculate RRF scores
        item_scores: dict[str, float] = {}
        item_objects: dict[str, RetrievedItem] = {}

        for weight, ranked_list in zip(weights, ranked_lists, strict=False):
            for rank, item in enumerate(ranked_list, 1):
                # Create unique key for item
                item_key = self._get_item_key(item)

                # RRF score formula: 1 / (k + rank)
                rrf_score = weight * (1 / (self.k + rank))

                if item_key in item_scores:
                    item_scores[item_key] += rrf_score
                else:
                    item_scores[item_key] = rrf_score
                    item_objects[item_key] = item

        # Sort by combined score
        sorted_keys = sorted(item_scores.keys(), key=lambda k: item_scores[k], reverse=True)

        # Update item scores and return
        result = []
        for key in sorted_keys:
            item = item_objects[key]
            item.score = item_scores[key]
            result.append(item)

        return result

    def _get_item_key(self, item: RetrievedItem) -> str:
        """Generate unique key for an item."""
        content_str = str(item.content)
        return hashlib.md5(content_str.encode()).hexdigest()

    def normalize_scores(self, items: list[RetrievedItem]) -> list[RetrievedItem]:
        """Normalize scores to 0-1 range."""
        if not items:
            return items

        max_score = max(item.score for item in items)
        min_score = min(item.score for item in items)

        if max_score == min_score:
            for item in items:
                item.score = 1.0
        else:
            for item in items:
                item.score = (item.score - min_score) / (max_score - min_score)

        return items


# =============================================================================
# Context Cache
# =============================================================================


class ContextCache:
    """Simple in-memory cache for retrieval results."""

    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        self._cache: dict[str, tuple[Any, float]] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds

    def _make_key(self, *args) -> str:
        """Create cache key from arguments."""
        return hashlib.md5(str(args).encode()).hexdigest()

    def get(self, *args) -> Any | None:
        """Get item from cache if not expired."""
        key = self._make_key(*args)
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self.ttl_seconds:
                return value
            else:
                del self._cache[key]
        return None

    def set(self, value: Any, *args) -> None:
        """Set item in cache."""
        if len(self._cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]

        key = self._make_key(*args)
        self._cache[key] = (value, time.time())

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()


# =============================================================================
# RAG Service Class
# =============================================================================


class RAGService:
    """
    RAG (Retrieval-Augmented Generation) Service for vehicle diagnostics.

    Features:
    - Multi-source context retrieval (Qdrant, Neo4j, PostgreSQL)
    - Hybrid ranking with Reciprocal Rank Fusion
    - Configurable LLM backend (Anthropic, OpenAI, Ollama)
    - Rule-based fallback when no LLM available
    - Hungarian language prompt templates
    - Confidence scoring based on retrieval quality
    - Async processing with caching
    """

    _instance: Optional["RAGService"] = None

    def __new__(cls) -> "RAGService":
        """Singleton pattern to reuse connections."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize RAG service."""
        if self._initialized:
            return

        self._initialized = True
        self._qdrant: QdrantService = qdrant_client
        self._embedding_service = None
        self._ranker = HybridRanker()
        self._cache = ContextCache()
        self._db_session: AsyncSession | None = None

        logger.info("RAGService initialized")

    def set_db_session(self, session: AsyncSession) -> None:
        """Set the database session for PostgreSQL queries."""
        self._db_session = session

    def _get_embedding_service(self):
        """Get embedding service instance."""
        if self._embedding_service is None:
            self._embedding_service = get_embedding_service()
        return self._embedding_service

    # =========================================================================
    # Retrieval Layer
    # =========================================================================

    async def retrieve_from_qdrant(
        self,
        query: str,
        collection: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        score_threshold: float = 0.5,
    ) -> list[RetrievedItem]:
        """
        Retrieve items from Qdrant vector store.

        Args:
            query: Search query text.
            collection: Qdrant collection name.
            top_k: Number of results to return.
            filters: Optional filter conditions.
            score_threshold: Minimum similarity score.

        Returns:
            List of RetrievedItem from Qdrant.
        """
        # Check cache
        cached = self._cache.get("qdrant", collection, query, filters)
        if cached is not None:
            return cached

        # Generate embedding for query
        query_embedding = embed_text(query, preprocess=True)

        try:
            results = await self._qdrant.search(
                collection_name=collection,
                query_vector=query_embedding,
                limit=top_k,
                filter_conditions=filters,
                score_threshold=score_threshold,
            )

            items = []
            for result in results:
                source = (RetrievalSource.QDRANT_DTC
                         if collection == QdrantService.DTC_COLLECTION
                         else RetrievalSource.QDRANT_SYMPTOM)

                items.append(RetrievedItem(
                    content=result.get("payload", {}),
                    source=source,
                    score=result.get("score", 0),
                    metadata={"id": result.get("id")},
                ))

            # Cache results
            self._cache.set(items, "qdrant", collection, query, filters)
            return items

        except Exception as e:
            logger.warning(f"Qdrant search error for {collection}: {e}")
            return []

    async def retrieve_from_neo4j(
        self,
        dtc_codes: list[str],
    ) -> tuple[list[RetrievedItem], dict[str, Any]]:
        """
        Retrieve graph context from Neo4j.

        Args:
            dtc_codes: List of DTC codes to look up.

        Returns:
            Tuple of (RetrievedItem list, combined graph data dict).
        """
        items = []
        combined_data = {
            "components": [],
            "repairs": [],
            "symptoms": [],
        }

        for code in dtc_codes:
            # Check cache
            cached = self._cache.get("neo4j", code)
            if cached is not None:
                path_data = cached
            else:
                try:
                    path_data = await get_diagnostic_path(code)
                    self._cache.set(path_data, "neo4j", code)
                except Exception as e:
                    logger.warning(f"Neo4j error for {code}: {e}")
                    path_data = {}

            if path_data:
                # Create item for DTC
                items.append(RetrievedItem(
                    content=path_data.get("dtc", {}),
                    source=RetrievalSource.NEO4J_GRAPH,
                    score=1.0,  # High confidence for direct match
                    metadata={"code": code},
                ))

                # Combine graph data
                combined_data["components"].extend(path_data.get("components", []))
                combined_data["repairs"].extend(path_data.get("repairs", []))
                combined_data["symptoms"].extend(path_data.get("symptoms", []))

        return items, combined_data

    async def retrieve_from_postgres(
        self,
        query: str,
        dtc_codes: list[str],
        vehicle_make: str | None = None,
        top_k: int = 10,
    ) -> list[RetrievedItem]:
        """
        Retrieve items from PostgreSQL using text search.

        Args:
            query: Search query text.
            dtc_codes: List of DTC codes for direct lookup.
            vehicle_make: Optional vehicle make filter.
            top_k: Number of results to return.

        Returns:
            List of RetrievedItem from PostgreSQL.
        """
        if self._db_session is None:
            logger.warning("No database session available for PostgreSQL search")
            return []

        items = []

        try:
            # Direct DTC code lookup
            if dtc_codes:
                stmt = select(DTCCode).where(
                    DTCCode.code.in_([c.upper() for c in dtc_codes])
                )
                result = await self._db_session.execute(stmt)
                dtc_records = result.scalars().all()

                for dtc in dtc_records:
                    items.append(RetrievedItem(
                        content={
                            "code": dtc.code,
                            "description": dtc.description_hu or dtc.description_en,
                            "description_hu": dtc.description_hu,
                            "description_en": dtc.description_en,
                            "category": dtc.category,
                            "severity": dtc.severity,
                            "symptoms": dtc.symptoms,
                            "possible_causes": dtc.possible_causes,
                            "diagnostic_steps": dtc.diagnostic_steps,
                        },
                        source=RetrievalSource.POSTGRES_TEXT,
                        score=1.0,  # Direct match
                        metadata={"id": dtc.id},
                    ))

            # Text search in DTC descriptions and symptoms
            if query:
                # Use ILIKE for simple text matching
                # In production, use PostgreSQL full-text search with tsvector
                search_stmt = select(DTCCode).where(
                    or_(
                        DTCCode.description_en.ilike(f"%{query}%"),
                        DTCCode.description_hu.ilike(f"%{query}%"),
                        func.array_to_string(DTCCode.symptoms, ' ').ilike(f"%{query}%"),
                    )
                ).limit(top_k)

                result = await self._db_session.execute(search_stmt)
                search_records = result.scalars().all()

                for dtc in search_records:
                    # Skip if already added via direct lookup
                    if any(i.content.get("code") == dtc.code for i in items):
                        continue

                    items.append(RetrievedItem(
                        content={
                            "code": dtc.code,
                            "description": dtc.description_hu or dtc.description_en,
                            "description_hu": dtc.description_hu,
                            "description_en": dtc.description_en,
                            "category": dtc.category,
                            "severity": dtc.severity,
                            "symptoms": dtc.symptoms,
                            "possible_causes": dtc.possible_causes,
                            "diagnostic_steps": dtc.diagnostic_steps,
                        },
                        source=RetrievalSource.POSTGRES_TEXT,
                        score=0.7,  # Lower score for text match
                        metadata={"id": dtc.id},
                    ))

            # Search known issues
            if query or vehicle_make:
                issue_stmt = select(KnownIssue)

                conditions = []
                if query:
                    conditions.append(
                        or_(
                            KnownIssue.title.ilike(f"%{query}%"),
                            KnownIssue.description.ilike(f"%{query}%"),
                        )
                    )
                if vehicle_make:
                    conditions.append(
                        KnownIssue.applicable_makes.any(vehicle_make)
                    )

                if conditions:
                    issue_stmt = issue_stmt.where(or_(*conditions)).limit(top_k)
                    result = await self._db_session.execute(issue_stmt)
                    issue_records = result.scalars().all()

                    for issue in issue_records:
                        items.append(RetrievedItem(
                            content={
                                "title": issue.title,
                                "description": issue.description,
                                "symptoms": issue.symptoms,
                                "causes": issue.causes,
                                "solutions": issue.solutions,
                                "related_dtc_codes": issue.related_dtc_codes,
                            },
                            source=RetrievalSource.POSTGRES_TEXT,
                            score=issue.confidence,
                            metadata={"id": issue.id, "type": "known_issue"},
                        ))

        except Exception as e:
            logger.error(f"PostgreSQL search error: {e}")

        return items

    # =========================================================================
    # Context Assembly
    # =========================================================================

    async def assemble_context(
        self,
        vehicle_info: VehicleInfo,
        dtc_codes: list[str],
        symptoms: str,
        recalls: list[dict[str, Any]] | None = None,
        complaints: list[dict[str, Any]] | None = None,
    ) -> RAGContext:
        """
        Assemble complete context from all sources.

        Performs parallel retrieval from multiple sources and combines
        results using hybrid ranking.

        Args:
            vehicle_info: Vehicle information.
            dtc_codes: List of DTC codes.
            symptoms: Symptom description in Hungarian.
            recalls: Optional NHTSA recalls.
            complaints: Optional NHTSA complaints.

        Returns:
            RAGContext with all retrieved and formatted context.
        """
        context = RAGContext()

        # Preprocess symptoms for search
        preprocessed_symptoms = preprocess_hungarian(symptoms) if symptoms else ""

        # Build search query combining symptoms and DTC codes
        search_query = f"{' '.join(dtc_codes)} {preprocessed_symptoms}".strip()

        # Parallel retrieval from all sources
        tasks = [
            # Qdrant DTC search
            self.retrieve_from_qdrant(
                query=search_query,
                collection=QdrantService.DTC_COLLECTION,
                top_k=10,
            ),
            # Qdrant symptom search
            self.retrieve_from_qdrant(
                query=preprocessed_symptoms or search_query,
                collection=QdrantService.SYMPTOM_COLLECTION,
                top_k=5,
                filters={"vehicle_make": vehicle_info.make} if vehicle_info.make else None,
            ),
            # Neo4j graph retrieval
            self.retrieve_from_neo4j(dtc_codes),
            # PostgreSQL text search
            self.retrieve_from_postgres(
                query=preprocessed_symptoms,
                dtc_codes=dtc_codes,
                vehicle_make=vehicle_info.make,
            ),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        qdrant_dtc_items = results[0] if not isinstance(results[0], Exception) else []
        qdrant_symptom_items = results[1] if not isinstance(results[1], Exception) else []
        neo4j_result = results[2] if not isinstance(results[2], Exception) else ([], {})
        postgres_items = results[3] if not isinstance(results[3], Exception) else []

        neo4j_items, graph_data = neo4j_result if isinstance(neo4j_result, tuple) else ([], {})

        # Store raw items
        context.dtc_items = qdrant_dtc_items
        context.symptom_items = qdrant_symptom_items
        context.graph_items = neo4j_items
        context.text_items = postgres_items
        context.graph_data = graph_data

        # Process recalls and complaints
        if recalls:
            for recall in recalls:
                context.recall_items.append(RetrievedItem(
                    content=recall,
                    source=RetrievalSource.NHTSA,
                    score=0.95,  # High relevance for recalls
                    metadata={"type": "recall"},
                ))

        if complaints:
            for complaint in complaints[:5]:  # Limit complaints
                context.recall_items.append(RetrievedItem(
                    content=complaint,
                    source=RetrievalSource.NHTSA,
                    score=0.6,
                    metadata={"type": "complaint"},
                ))

        # Hybrid ranking of DTC results
        all_dtc_lists = [context.dtc_items, context.text_items]
        combined_dtc = self._ranker.reciprocal_rank_fusion(
            [items for items in all_dtc_lists if items],
            weights=[1.0, 0.8]
        )

        # Format context strings
        dtc_data = [item.content for item in combined_dtc[:10]]
        context.dtc_context = format_dtc_context(dtc_data)

        symptom_data = [
            {
                "description": item.content.get("description", ""),
                "score": item.score,
                "resolution": item.content.get("resolution", ""),
                "related_dtc": item.content.get("related_dtc", []),
            }
            for item in context.symptom_items[:5]
        ]
        context.symptom_context = format_symptom_context(symptom_data)

        context.repair_context = format_repair_context(graph_data)

        recall_data = [item.content for item in context.recall_items if item.metadata.get("type") == "recall"]
        complaint_data = [item.content for item in context.recall_items if item.metadata.get("type") == "complaint"]
        context.recall_context = format_recall_context(recall_data, complaint_data)

        return context

    # =========================================================================
    # Confidence Scoring
    # =========================================================================

    def calculate_confidence(
        self,
        context: RAGContext,
        dtc_codes: list[str],
    ) -> tuple[ConfidenceLevel, float]:
        """
        Calculate confidence score for diagnosis.

        Args:
            context: Assembled RAG context.
            dtc_codes: Input DTC codes.

        Returns:
            Tuple of (ConfidenceLevel, score between 0.0 and 1.0).
        """
        score = 0.0
        factors = 0

        # Factor 1: DTC context quality (0-0.3)
        if context.dtc_items:
            dtc_scores = [item.score for item in context.dtc_items]
            if dtc_scores:
                avg_score = sum(dtc_scores) / len(dtc_scores)
                score += avg_score * 0.3
            factors += 0.3

        # Factor 2: PostgreSQL direct matches (0-0.2)
        direct_matches = sum(1 for item in context.text_items if item.score >= 1.0)
        if dtc_codes:
            match_ratio = direct_matches / len(dtc_codes)
            score += match_ratio * 0.2
            factors += 0.2

        # Factor 3: Symptom matching (0-0.2)
        if context.symptom_items:
            symptom_scores = [item.score for item in context.symptom_items]
            if symptom_scores:
                avg_symptom = sum(symptom_scores) / len(symptom_scores)
                score += avg_symptom * 0.2
            factors += 0.2

        # Factor 4: Graph context richness (0-0.2)
        if context.graph_data:
            graph_score = 0.0
            if context.graph_data.get("components"):
                graph_score += 0.08
            if context.graph_data.get("repairs"):
                graph_score += 0.08
            if context.graph_data.get("symptoms"):
                graph_score += 0.04
            score += graph_score
            factors += 0.2

        # Factor 5: NHTSA data available (0-0.1)
        if context.recall_items:
            score += 0.1
            factors += 0.1

        # Normalize score
        normalized_score = min(1.0, score / factors) if factors > 0 else 0.1

        # Determine confidence level
        if normalized_score >= 0.75:
            level = ConfidenceLevel.HIGH
        elif normalized_score >= 0.5:
            level = ConfidenceLevel.MEDIUM
        elif normalized_score >= 0.25:
            level = ConfidenceLevel.LOW
        else:
            level = ConfidenceLevel.UNKNOWN

        return level, normalized_score

    # =========================================================================
    # Response Generation
    # =========================================================================

    async def generate_diagnosis(
        self,
        vehicle_info: VehicleInfo,
        dtc_codes: list[str],
        symptoms: str,
        context: RAGContext,
        additional_context: str | None = None,
    ) -> ParsedDiagnosisResponse:
        """
        Generate diagnosis using LLM with assembled context.

        Falls back to rule-based diagnosis if no LLM is available.

        Args:
            vehicle_info: Vehicle information.
            dtc_codes: List of DTC codes.
            symptoms: Symptom description.
            context: Assembled RAG context.
            additional_context: Optional additional context from user.

        Returns:
            ParsedDiagnosisResponse with diagnosis results.
        """
        # Build prompt context
        prompt_context = DiagnosisPromptContext(
            make=vehicle_info.make,
            model=vehicle_info.model,
            year=vehicle_info.year,
            engine_code=vehicle_info.engine_code,
            mileage_km=vehicle_info.mileage_km,
            vin=vehicle_info.vin,
            dtc_codes=dtc_codes,
            symptoms=symptoms,
            additional_context=additional_context,
            dtc_context=context.dtc_context,
            symptom_context=context.symptom_context,
            repair_context=context.repair_context,
            recall_context=context.recall_context,
        )

        user_prompt = build_diagnosis_prompt(prompt_context)

        # Check if LLM is available
        if not is_llm_available():
            logger.info("No LLM available, using rule-based diagnosis")
            # Prepare DTC data for rule-based diagnosis
            dtc_data = [item.content for item in context.dtc_items + context.text_items]
            recall_data = [item.content for item in context.recall_items
                         if item.metadata.get("type") == "recall"]
            complaint_data = [item.content for item in context.recall_items
                            if item.metadata.get("type") == "complaint"]

            return generate_rule_based_diagnosis(
                dtc_codes=dtc_data,
                vehicle_info=vehicle_info.to_dict(),
                recalls=recall_data,
                complaints=complaint_data,
            )

        # Get LLM provider and generate
        try:
            provider = get_llm_provider()
            llm_config = LLMConfig(
                temperature=0.3,
                max_tokens=4096,
            )

            response = await provider.generate_with_system(
                system_prompt=SYSTEM_PROMPT_HU,
                user_prompt=user_prompt,
                config=llm_config,
            )

            # Parse the response
            parsed = parse_diagnosis_response(response.content)

            # Add metadata
            parsed.raw_response = response.content

            return parsed

        except Exception as e:
            logger.error(f"LLM generation error: {e}, falling back to rule-based")

            # Fallback to rule-based
            dtc_data = [item.content for item in context.dtc_items + context.text_items]
            recall_data = [item.content for item in context.recall_items
                         if item.metadata.get("type") == "recall"]

            result = generate_rule_based_diagnosis(
                dtc_codes=dtc_data,
                vehicle_info=vehicle_info.to_dict(),
                recalls=recall_data,
            )
            result.parse_error = f"LLM error: {e!s}"
            return result

    # =========================================================================
    # Main Diagnosis Method
    # =========================================================================

    async def diagnose(
        self,
        vehicle_info: dict[str, Any],
        dtc_codes: list[str],
        symptoms: str,
        recalls: list[dict[str, Any]] | None = None,
        complaints: list[dict[str, Any]] | None = None,
        additional_context: str | None = None,
    ) -> DiagnosisResult:
        """
        Perform complete vehicle diagnosis using RAG.

        This is the main entry point for the diagnosis pipeline.

        Args:
            vehicle_info: Dictionary with vehicle information.
            dtc_codes: List of DTC codes.
            symptoms: Free-text description of symptoms in Hungarian.
            recalls: Optional NHTSA recalls.
            complaints: Optional NHTSA complaints.
            additional_context: Optional additional context.

        Returns:
            DiagnosisResult with complete diagnosis information.
        """
        start_time = time.time()

        # Parse vehicle info
        vehicle = VehicleInfo(
            make=vehicle_info.get("make", ""),
            model=vehicle_info.get("model", ""),
            year=vehicle_info.get("year", 0),
            vin=vehicle_info.get("vin"),
            engine_code=vehicle_info.get("engine_code"),
            mileage_km=vehicle_info.get("mileage_km"),
        )

        # Normalize DTC codes
        dtc_codes = [code.upper().strip() for code in dtc_codes if code.strip()]

        logger.info(
            f"Starting RAG diagnosis for {vehicle.make} {vehicle.model} {vehicle.year}, "
            f"DTCs: {dtc_codes}"
        )

        # Assemble context from all sources
        context = await self.assemble_context(
            vehicle_info=vehicle,
            dtc_codes=dtc_codes,
            symptoms=symptoms,
            recalls=recalls,
            complaints=complaints,
        )

        # Calculate confidence
        confidence_level, confidence_score = self.calculate_confidence(context, dtc_codes)

        # Generate diagnosis
        diagnosis = await self.generate_diagnosis(
            vehicle_info=vehicle,
            dtc_codes=dtc_codes,
            symptoms=symptoms,
            context=context,
            additional_context=additional_context,
        )

        # Determine provider info
        provider = get_llm_provider()
        model_name = provider.model_name
        provider_name = provider.provider_type.value

        used_fallback = (provider.provider_type == LLMProviderType.RULE_BASED or
                        diagnosis.parse_error is not None)

        # Build repair recommendations
        repair_recommendations = []
        for repair in diagnosis.recommended_repairs:
            repair_recommendations.append(RepairRecommendation(
                name=repair.get("title", ""),
                description=repair.get("description", ""),
                difficulty=repair.get("difficulty", "intermediate"),
                estimated_time_minutes=repair.get("estimated_time_minutes"),
                estimated_cost_min=repair.get("estimated_cost_min"),
                estimated_cost_max=repair.get("estimated_cost_max"),
                parts=repair.get("parts_needed", []),
            ))

        # Build sources
        sources = []
        seen_sources = set()

        for item in context.get_all_items()[:10]:
            source_name = f"{item.source.value}"
            if source_name not in seen_sources:
                sources.append({
                    "type": item.source.value,
                    "title": item.content.get("code") or item.content.get("title", source_name),
                    "relevance_score": item.score,
                })
                seen_sources.add(source_name)

        processing_time = int((time.time() - start_time) * 1000)

        # Combine confidence from context and diagnosis
        final_confidence = max(confidence_score, diagnosis.confidence_score) if diagnosis.probable_causes else confidence_score

        result = DiagnosisResult(
            dtc_codes=dtc_codes,
            symptoms=symptoms,
            vehicle_info=vehicle,
            diagnosis_summary=diagnosis.summary,
            root_cause_analysis="\n".join(
                f"- {cause.get('title', '')}: {cause.get('description', '')}"
                for cause in diagnosis.probable_causes[:3]
            ),
            confidence=confidence_level,
            confidence_score=final_confidence,
            probable_causes=diagnosis.probable_causes,
            repair_recommendations=repair_recommendations,
            safety_warnings=diagnosis.safety_warnings,
            diagnostic_steps=diagnosis.diagnostic_steps,
            similar_cases=[item.content for item in context.symptom_items[:5]],
            related_dtc_info=[item.content for item in context.dtc_items[:10]],
            sources=sources,
            model_used=model_name,
            provider_used=provider_name,
            processing_time_ms=processing_time,
            used_fallback=used_fallback,
        )

        logger.info(
            f"Diagnosis completed in {processing_time}ms, "
            f"provider: {provider_name}, "
            f"confidence: {confidence_level.value} ({final_confidence:.2%})"
        )

        return result

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def warmup(self) -> None:
        """Warm up service by initializing components."""
        logger.info("Warming up RAG service...")
        self._get_embedding_service().warmup()

        # Check LLM availability
        if is_llm_available():
            provider = get_llm_provider()
            logger.info(f"LLM provider ready: {provider.provider_type.value}")
        else:
            logger.warning("No LLM API available, will use rule-based fallback")

        logger.info("RAG service warmup complete")

    def clear_cache(self) -> None:
        """Clear the retrieval cache."""
        self._cache.clear()
        logger.info("RAG cache cleared")


# =============================================================================
# Module-level Functions
# =============================================================================


_rag_service: RAGService | None = None


def get_rag_service() -> RAGService:
    """
    Get the global RAG service instance.

    Returns:
        RAGService: The singleton RAG service instance.
    """
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


async def diagnose(
    vehicle_info: dict[str, Any],
    dtc_codes: list[str],
    symptoms: str,
    recalls: list[dict[str, Any]] | None = None,
    complaints: list[dict[str, Any]] | None = None,
    db_session: AsyncSession | None = None,
) -> DiagnosisResult:
    """
    Convenience function to perform vehicle diagnosis.

    Args:
        vehicle_info: Dictionary with vehicle information.
        dtc_codes: List of DTC codes.
        symptoms: Free-text description of symptoms.
        recalls: Optional NHTSA recalls.
        complaints: Optional NHTSA complaints.
        db_session: Optional database session for PostgreSQL queries.

    Returns:
        DiagnosisResult with complete diagnosis information.
    """
    service = get_rag_service()

    if db_session:
        service.set_db_session(db_session)

    return await service.diagnose(
        vehicle_info=vehicle_info,
        dtc_codes=dtc_codes,
        symptoms=symptoms,
        recalls=recalls,
        complaints=complaints,
    )


async def get_context(
    query: str,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """
    Convenience function to retrieve context from vector store.

    Args:
        query: Search query text.
        top_k: Number of results to return.

    Returns:
        List of matching documents with scores.
    """
    service = get_rag_service()
    items = await service.retrieve_from_qdrant(
        query=query,
        collection=QdrantService.DTC_COLLECTION,
        top_k=top_k,
    )
    return [
        {
            "content": item.content,
            "score": item.score,
            "source": item.source.value,
        }
        for item in items
    ]
