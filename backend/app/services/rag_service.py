"""
RAG (Retrieval-Augmented Generation) Service for AutoCognitix.

This module provides Hungarian vehicle diagnostic RAG functionality using:
- LangChain for LLM orchestration
- Qdrant for vector similarity search
- Neo4j for graph-based knowledge enrichment
- Hungarian embedding service for semantic search
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.core.config import settings
from app.core.logging import get_logger
from app.db.neo4j_models import get_diagnostic_path, DTCNode, VehicleNode
from app.db.qdrant_client import qdrant_client, QdrantService
from app.services.embedding_service import (
    embed_text,
    embed_batch,
    get_embedding_service,
)

logger = get_logger(__name__)


# =============================================================================
# Enums and Data Classes
# =============================================================================


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"


class ConfidenceLevel(str, Enum):
    """Confidence levels for diagnosis."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


@dataclass
class VehicleInfo:
    """Vehicle information for diagnosis."""
    make: str
    model: str
    year: int
    vin: Optional[str] = None
    engine_code: Optional[str] = None
    mileage_km: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "make": self.make,
            "model": self.model,
            "year": self.year,
            "vin": self.vin,
            "engine_code": self.engine_code,
            "mileage_km": self.mileage_km,
        }


@dataclass
class RepairRecommendation:
    """Repair recommendation with cost and time estimates."""
    name: str
    description: str
    difficulty: str
    estimated_time_minutes: Optional[int] = None
    estimated_cost_min: Optional[int] = None
    estimated_cost_max: Optional[int] = None
    parts: List[Dict[str, Any]] = field(default_factory=list)
    diagnostic_steps: List[str] = field(default_factory=list)


@dataclass
class DiagnosisResult:
    """Complete diagnosis result with confidence and recommendations."""
    dtc_codes: List[str]
    symptoms: str
    vehicle_info: VehicleInfo

    # Main diagnosis
    diagnosis_summary: str
    root_cause_analysis: str
    confidence: ConfidenceLevel
    confidence_score: float  # 0.0 - 1.0

    # Recommendations
    repair_recommendations: List[RepairRecommendation] = field(default_factory=list)
    safety_warnings: List[str] = field(default_factory=list)

    # Context used
    similar_cases: List[Dict[str, Any]] = field(default_factory=list)
    related_dtc_info: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    generated_at: datetime = field(default_factory=datetime.utcnow)
    model_used: str = ""
    processing_time_ms: int = 0


@dataclass
class RAGContext:
    """Context assembled for RAG generation."""
    dtc_context: List[Dict[str, Any]] = field(default_factory=list)
    symptom_context: List[Dict[str, Any]] = field(default_factory=list)
    graph_context: Dict[str, Any] = field(default_factory=dict)
    vehicle_specific: Dict[str, Any] = field(default_factory=dict)
    recalls: List[Dict[str, Any]] = field(default_factory=list)

    def to_formatted_string(self) -> str:
        """Format context for LLM prompt."""
        sections = []

        # DTC context
        if self.dtc_context:
            dtc_section = "## Kapcsolodo DTC informaciok:\n"
            for dtc in self.dtc_context:
                dtc_section += f"- **{dtc.get('code', 'N/A')}**: {dtc.get('description', 'N/A')}\n"
                dtc_section += f"  Sulyossag: {dtc.get('severity', 'N/A')}, Kategoria: {dtc.get('category', 'N/A')}\n"
            sections.append(dtc_section)

        # Symptom context
        if self.symptom_context:
            symptom_section = "## Hasonlo tunetek korabbi esetekbol:\n"
            for idx, symptom in enumerate(self.symptom_context[:5], 1):
                symptom_section += f"{idx}. {symptom.get('description', 'N/A')} "
                symptom_section += f"(hasonlosag: {symptom.get('score', 0):.2%})\n"
            sections.append(symptom_section)

        # Graph context (components, repairs)
        if self.graph_context:
            if self.graph_context.get("components"):
                comp_section = "## Erintett komponensek:\n"
                for comp in self.graph_context["components"]:
                    comp_section += f"- {comp.get('name', 'N/A')} ({comp.get('system', 'N/A')})\n"
                sections.append(comp_section)

            if self.graph_context.get("repairs"):
                repair_section = "## Lehetseges javitasok:\n"
                for repair in self.graph_context["repairs"]:
                    repair_section += f"- **{repair.get('name', 'N/A')}**\n"
                    repair_section += f"  Nehezseg: {repair.get('difficulty', 'N/A')}, "
                    repair_section += f"Ido: {repair.get('estimated_time_minutes', 'N/A')} perc\n"
                    if repair.get("parts"):
                        repair_section += f"  Alkatreszek: {', '.join(p.get('name', '') for p in repair['parts'])}\n"
                sections.append(repair_section)

        # Recalls
        if self.recalls:
            recall_section = "## Aktiv visszahivasok:\n"
            for recall in self.recalls:
                recall_section += f"- {recall.get('component', 'N/A')}: {recall.get('summary', 'N/A')[:100]}...\n"
            sections.append(recall_section)

        return "\n".join(sections) if sections else "Nincs elérhető kontextus."


# =============================================================================
# Hungarian Prompt Templates
# =============================================================================


HUNGARIAN_DIAGNOSIS_TEMPLATE = """Te egy tapasztalt magyar gepjarmu-diagnosztikai szakerto vagy.
A feladatod, hogy alapos es ertelmezheto diagnosztikai elemzest adj a jarmu problemairol.

## Jarmu adatok:
- Gyarto: {make}
- Modell: {model}
- Evjarat: {year}
- Kilometerora: {mileage_km} km
- Motorkod: {engine_code}
- VIN: {vin}

## Bejelentett hibakodok (DTC):
{dtc_codes}

## Ugyfel altal leirt tunetek:
{symptoms}

## Kapcsolodo informaciok az adatbazisbol:
{context}

---

Kerlek, keszits reszletes diagnosztikai elemzest az alabbi struktura szerint:

### 1. OSSZEFOGLALO
Rovid, erthetoo osszefoglaloja a problemanak (max 2-3 mondat).

### 2. LEHETSEGES OKOK
Sorold fel a lehetseges hibaokok-at valoszinuseguk sorrendjeben:
1. [Legvaloszinubb ok] - [magyarazat]
2. [Masodik legvaloszinubb ok] - [magyarazat]
3. stb.

### 3. JAVASOLT ELLENORZESEK
Milyen diagnosztikai lepeseket kell elvegezni a pontos hibaokok meghatarozasahoz:
1. [Ellenorzes 1]
2. [Ellenorzes 2]
stb.

### 4. JAVITASI JAVASLATOK
Ajanlott javitasi muveletek prioritas sorrendben:
- [Javitas 1]: [becsult koltseg es munkaido]
- [Javitas 2]: [becsult koltseg es munkaido]

### 5. BIZTONSAGI FIGYELMEZTETESEK
Ha vannak biztonsagi kockazatok, sorold fel oket.

### 6. MEGJEGYZESEK
Egyeb fontos informaciok, tanacs a jarmu tulajdonosnak.

Valaszolj magyarul, szakmailag pontosan, de erthetoen a laikusok szamara is!"""


CONFIDENCE_ASSESSMENT_TEMPLATE = """Ertekeld a kovetkezo diagnosztikai kontextus megbizhatosagat 0 es 1 kozotti skalan:

Kontextus:
- DTC kodok szama: {dtc_count}
- Talalat a DTC adatbazisban: {dtc_matches}
- Hasonlo tunetek szama: {symptom_matches}
- Graf kapcsolatok szama: {graph_connections}
- Jarmu-specifikus adatok: {vehicle_specific}

Valaszolj CSAK egy szammal 0.0 es 1.0 kozott, ahol:
- 0.0-0.3: Alacsony megbizhatosag (kevés vagy ellentmondásos adat)
- 0.3-0.6: Kozepes megbizhatosag (reszleges egyezes)
- 0.6-0.8: Jo megbizhatosag (tobb forras egyezik)
- 0.8-1.0: Magas megbizhatosag (erős egyezés több forrásból)

Konfidencia score:"""


# =============================================================================
# RAG Service Class
# =============================================================================


class RAGService:
    """
    RAG (Retrieval-Augmented Generation) Service for vehicle diagnostics.

    Features:
    - Multi-source context retrieval (Qdrant vectors, Neo4j graph)
    - Configurable LLM backend (Anthropic, OpenAI, Ollama)
    - Hungarian language prompt templates
    - Confidence scoring
    - Async processing
    """

    _instance: Optional["RAGService"] = None

    def __new__(cls) -> "RAGService":
        """Singleton pattern to reuse LLM connections."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize RAG service."""
        if self._initialized:
            return

        self._initialized = True
        self._llm = None
        self._embedding_service = None
        self._qdrant: QdrantService = qdrant_client

        logger.info(f"RAGService initialized with LLM provider: {settings.LLM_PROVIDER}")

    def _get_llm(self):
        """Get or create LLM instance based on configuration."""
        if self._llm is not None:
            return self._llm

        provider = LLMProvider(settings.LLM_PROVIDER.lower())

        if provider == LLMProvider.ANTHROPIC:
            from langchain_anthropic import ChatAnthropic

            self._llm = ChatAnthropic(
                model=settings.ANTHROPIC_MODEL,
                anthropic_api_key=settings.ANTHROPIC_API_KEY,
                temperature=0.3,
                max_tokens=4096,
            )
            logger.info(f"Using Anthropic model: {settings.ANTHROPIC_MODEL}")

        elif provider == LLMProvider.OPENAI:
            from langchain_openai import ChatOpenAI

            self._llm = ChatOpenAI(
                model=settings.OPENAI_MODEL,
                openai_api_key=settings.OPENAI_API_KEY,
                temperature=0.3,
                max_tokens=4096,
            )
            logger.info(f"Using OpenAI model: {settings.OPENAI_MODEL}")

        elif provider == LLMProvider.OLLAMA:
            from langchain_community.llms import Ollama

            self._llm = Ollama(
                model=settings.OLLAMA_MODEL,
                base_url=settings.OLLAMA_BASE_URL,
                temperature=0.3,
            )
            logger.info(f"Using Ollama model: {settings.OLLAMA_MODEL}")

        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        return self._llm

    def _get_embedding_service(self):
        """Get embedding service instance."""
        if self._embedding_service is None:
            self._embedding_service = get_embedding_service()
        return self._embedding_service

    # =========================================================================
    # Context Retrieval
    # =========================================================================

    async def get_context(
        self,
        query: str,
        top_k: int = 10,
        collection_name: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context from Qdrant vector store.

        Args:
            query: Search query text
            top_k: Number of results to return
            collection_name: Specific collection to search (default: search all)
            filters: Optional filter conditions

        Returns:
            List of matching documents with scores
        """
        # Generate embedding for query
        query_embedding = embed_text(query, preprocess=True)

        results = []

        # Determine which collections to search
        if collection_name:
            collections = [collection_name]
        else:
            collections = [
                QdrantService.DTC_COLLECTION,
                QdrantService.SYMPTOM_COLLECTION,
                QdrantService.ISSUE_COLLECTION,
            ]

        # Search each collection
        for coll in collections:
            try:
                search_results = await self._qdrant.search(
                    collection_name=coll,
                    query_vector=query_embedding,
                    limit=top_k,
                    filter_conditions=filters,
                    score_threshold=0.5,  # Minimum similarity threshold
                )

                for result in search_results:
                    result["collection"] = coll
                    results.append(result)

            except Exception as e:
                logger.warning(f"Error searching collection {coll}: {e}")
                continue

        # Sort by score and limit
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return results[:top_k]

    async def _get_dtc_context(
        self,
        dtc_codes: List[str],
        vehicle_make: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve context for specific DTC codes."""
        context = []

        for code in dtc_codes:
            # Search Qdrant for similar DTCs
            query_text = f"DTC hibakod {code}"
            embedding = embed_text(query_text)

            try:
                results = await self._qdrant.search_dtc(
                    query_vector=embedding,
                    limit=5,
                )

                for result in results:
                    payload = result.get("payload", {})
                    context.append({
                        "code": payload.get("code", code),
                        "description": payload.get("description_hu") or payload.get("description", ""),
                        "severity": payload.get("severity", "unknown"),
                        "category": payload.get("category", "unknown"),
                        "score": result.get("score", 0),
                    })

            except Exception as e:
                logger.warning(f"Error getting DTC context for {code}: {e}")

            # Also get graph context from Neo4j
            try:
                graph_data = await get_diagnostic_path(code)
                if graph_data:
                    context.append({
                        "code": code,
                        "description": graph_data.get("dtc", {}).get("description", ""),
                        "severity": graph_data.get("dtc", {}).get("severity", ""),
                        "symptoms": graph_data.get("symptoms", []),
                        "components": graph_data.get("components", []),
                        "repairs": graph_data.get("repairs", []),
                        "source": "neo4j_graph",
                    })
            except Exception as e:
                logger.warning(f"Error getting graph context for {code}: {e}")

        return context

    async def _get_symptom_context(
        self,
        symptoms: str,
        vehicle_make: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve context for symptom description."""
        if not symptoms or not symptoms.strip():
            return []

        embedding = embed_text(symptoms, preprocess=True)

        try:
            results = await self._qdrant.search_similar_symptoms(
                query_vector=embedding,
                limit=top_k,
                vehicle_make=vehicle_make,
            )

            context = []
            for result in results:
                payload = result.get("payload", {})
                context.append({
                    "description": payload.get("description", ""),
                    "related_dtc": payload.get("related_dtc", []),
                    "vehicle_make": payload.get("vehicle_make", ""),
                    "resolution": payload.get("resolution", ""),
                    "score": result.get("score", 0),
                })

            return context

        except Exception as e:
            logger.warning(f"Error getting symptom context: {e}")
            return []

    async def _get_graph_context(
        self,
        dtc_codes: List[str],
        vehicle_info: VehicleInfo,
    ) -> Dict[str, Any]:
        """Retrieve comprehensive graph context from Neo4j."""
        combined_context = {
            "components": [],
            "repairs": [],
            "symptoms": [],
        }

        for code in dtc_codes:
            try:
                path_data = await get_diagnostic_path(code)
                if path_data:
                    combined_context["components"].extend(
                        path_data.get("components", [])
                    )
                    combined_context["repairs"].extend(
                        path_data.get("repairs", [])
                    )
                    combined_context["symptoms"].extend(
                        path_data.get("symptoms", [])
                    )
            except Exception as e:
                logger.warning(f"Error getting graph context for {code}: {e}")

        # Check for vehicle-specific common issues
        try:
            vehicle_node = VehicleNode.nodes.filter(
                make=vehicle_info.make,
                model=vehicle_info.model,
            ).first()

            if vehicle_node:
                for dtc in vehicle_node.has_common_issue.all():
                    rel = vehicle_node.has_common_issue.relationship(dtc)
                    if rel.year_start <= vehicle_info.year <= (rel.year_end or 9999):
                        combined_context["vehicle_specific_issues"] = {
                            "dtc": dtc.code,
                            "frequency": rel.frequency,
                        }
        except Exception as e:
            logger.debug(f"No vehicle-specific data found: {e}")

        return combined_context

    async def _assemble_context(
        self,
        vehicle_info: VehicleInfo,
        dtc_codes: List[str],
        symptoms: str,
    ) -> RAGContext:
        """Assemble complete context from all sources."""
        # Gather context from multiple sources in parallel
        dtc_task = self._get_dtc_context(dtc_codes, vehicle_info.make)
        symptom_task = self._get_symptom_context(symptoms, vehicle_info.make)
        graph_task = self._get_graph_context(dtc_codes, vehicle_info)

        dtc_context, symptom_context, graph_context = await asyncio.gather(
            dtc_task, symptom_task, graph_task
        )

        return RAGContext(
            dtc_context=dtc_context,
            symptom_context=symptom_context,
            graph_context=graph_context,
        )

    # =========================================================================
    # Response Generation
    # =========================================================================

    async def generate_response(
        self,
        context: RAGContext,
        query: str,
        vehicle_info: Optional[VehicleInfo] = None,
    ) -> str:
        """
        Generate response using LLM with provided context.

        Args:
            context: Assembled RAG context
            query: User query or symptom description
            vehicle_info: Optional vehicle information

        Returns:
            Generated response text
        """
        llm = self._get_llm()

        # Build prompt with context
        prompt = ChatPromptTemplate.from_template(
            """A kovetkezo kontextus alapjan valaszolj a kerdesre magyarul:

Kontextus:
{context}

Kerdes/Problema leirasa:
{query}

Reszletes valasz:"""
        )

        # Create chain
        chain = prompt | llm | StrOutputParser()

        # Generate response
        response = await chain.ainvoke({
            "context": context.to_formatted_string(),
            "query": query,
        })

        return response

    # =========================================================================
    # Confidence Scoring
    # =========================================================================

    def _calculate_confidence(
        self,
        context: RAGContext,
        dtc_codes: List[str],
    ) -> Tuple[ConfidenceLevel, float]:
        """
        Calculate confidence score for diagnosis.

        Args:
            context: Assembled RAG context
            dtc_codes: Input DTC codes

        Returns:
            Tuple of (ConfidenceLevel, score between 0.0 and 1.0)
        """
        score = 0.0
        factors = 0

        # Factor 1: DTC context quality (0-0.3)
        if context.dtc_context:
            dtc_scores = [d.get("score", 0) for d in context.dtc_context if "score" in d]
            if dtc_scores:
                avg_dtc_score = sum(dtc_scores) / len(dtc_scores)
                score += avg_dtc_score * 0.3
            factors += 0.3

        # Factor 2: Symptom matching (0-0.25)
        if context.symptom_context:
            symptom_scores = [s.get("score", 0) for s in context.symptom_context]
            if symptom_scores:
                avg_symptom_score = sum(symptom_scores) / len(symptom_scores)
                score += avg_symptom_score * 0.25
            factors += 0.25

        # Factor 3: Graph context richness (0-0.25)
        if context.graph_context:
            graph_score = 0.0
            if context.graph_context.get("components"):
                graph_score += 0.1
            if context.graph_context.get("repairs"):
                graph_score += 0.1
            if context.graph_context.get("symptoms"):
                graph_score += 0.05
            score += graph_score
            factors += 0.25

        # Factor 4: DTC coverage (0-0.2)
        if dtc_codes:
            found_codes = set()
            for dtc in context.dtc_context:
                if dtc.get("code") in dtc_codes:
                    found_codes.add(dtc.get("code"))
            coverage = len(found_codes) / len(dtc_codes)
            score += coverage * 0.2
            factors += 0.2

        # Normalize score
        if factors > 0:
            normalized_score = min(1.0, score / factors)
        else:
            normalized_score = 0.1  # Minimal confidence if no context

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
    # Main Diagnosis Method
    # =========================================================================

    async def diagnose(
        self,
        vehicle_info: Dict[str, Any],
        dtc_codes: List[str],
        symptoms: str,
    ) -> DiagnosisResult:
        """
        Perform complete vehicle diagnosis using RAG.

        Args:
            vehicle_info: Dictionary with vehicle information (make, model, year, etc.)
            dtc_codes: List of DTC codes (e.g., ["P0300", "P0171"])
            symptoms: Free-text description of symptoms in Hungarian

        Returns:
            DiagnosisResult with complete diagnosis information
        """
        import time
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
            f"Starting diagnosis for {vehicle.make} {vehicle.model} {vehicle.year}, "
            f"DTCs: {dtc_codes}"
        )

        # Assemble context
        context = await self._assemble_context(vehicle, dtc_codes, symptoms)

        # Calculate confidence
        confidence_level, confidence_score = self._calculate_confidence(
            context, dtc_codes
        )

        # Generate diagnosis using LLM
        llm = self._get_llm()

        prompt = ChatPromptTemplate.from_template(HUNGARIAN_DIAGNOSIS_TEMPLATE)
        chain = prompt | llm | StrOutputParser()

        # Format DTC codes for prompt
        dtc_formatted = "\n".join([f"- {code}" for code in dtc_codes]) if dtc_codes else "Nincs hibakod"

        diagnosis_text = await chain.ainvoke({
            "make": vehicle.make,
            "model": vehicle.model,
            "year": vehicle.year,
            "mileage_km": vehicle.mileage_km or "N/A",
            "engine_code": vehicle.engine_code or "N/A",
            "vin": vehicle.vin or "N/A",
            "dtc_codes": dtc_formatted,
            "symptoms": symptoms or "Nincs leirva",
            "context": context.to_formatted_string(),
        })

        # Parse diagnosis sections
        diagnosis_summary = self._extract_section(diagnosis_text, "OSSZEFOGLALO", "LEHETSEGES OKOK")
        root_cause = self._extract_section(diagnosis_text, "LEHETSEGES OKOK", "JAVASOLT ELLENORZESEK")

        # Extract safety warnings
        safety_section = self._extract_section(diagnosis_text, "BIZTONSAGI FIGYELMEZTETESEK", "MEGJEGYZESEK")
        safety_warnings = [w.strip() for w in safety_section.split("\n") if w.strip() and w.strip() != "-"]

        # Build repair recommendations from graph context
        repair_recommendations = []
        for repair in context.graph_context.get("repairs", []):
            repair_recommendations.append(RepairRecommendation(
                name=repair.get("name", ""),
                description=repair.get("description", ""),
                difficulty=repair.get("difficulty", "intermediate"),
                estimated_time_minutes=repair.get("estimated_time_minutes"),
                estimated_cost_min=repair.get("estimated_cost_min"),
                estimated_cost_max=repair.get("estimated_cost_max"),
                parts=repair.get("parts", []),
                diagnostic_steps=repair.get("diagnostic_steps", []),
            ))

        processing_time = int((time.time() - start_time) * 1000)

        # Determine model name
        provider = LLMProvider(settings.LLM_PROVIDER.lower())
        if provider == LLMProvider.ANTHROPIC:
            model_name = settings.ANTHROPIC_MODEL
        elif provider == LLMProvider.OPENAI:
            model_name = settings.OPENAI_MODEL
        else:
            model_name = settings.OLLAMA_MODEL

        result = DiagnosisResult(
            dtc_codes=dtc_codes,
            symptoms=symptoms,
            vehicle_info=vehicle,
            diagnosis_summary=diagnosis_summary,
            root_cause_analysis=root_cause,
            confidence=confidence_level,
            confidence_score=confidence_score,
            repair_recommendations=repair_recommendations,
            safety_warnings=safety_warnings,
            similar_cases=context.symptom_context[:5],
            related_dtc_info=context.dtc_context,
            model_used=model_name,
            processing_time_ms=processing_time,
        )

        logger.info(
            f"Diagnosis completed in {processing_time}ms, "
            f"confidence: {confidence_level.value} ({confidence_score:.2%})"
        )

        return result

    def _extract_section(
        self,
        text: str,
        start_marker: str,
        end_marker: str,
    ) -> str:
        """Extract text section between markers."""
        try:
            start_idx = text.find(start_marker)
            if start_idx == -1:
                return ""

            start_idx = text.find("\n", start_idx) + 1

            end_idx = text.find(end_marker, start_idx)
            if end_idx == -1:
                end_idx = len(text)

            section = text[start_idx:end_idx].strip()

            # Remove markdown headers
            lines = section.split("\n")
            cleaned_lines = [
                line for line in lines
                if not line.strip().startswith("###")
            ]

            return "\n".join(cleaned_lines).strip()

        except Exception:
            return ""

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def warmup(self) -> None:
        """Warm up service by initializing LLM and embedding service."""
        logger.info("Warming up RAG service...")
        self._get_llm()
        self._get_embedding_service().warmup()
        logger.info("RAG service warmup complete")


# =============================================================================
# Module-level Functions
# =============================================================================


_rag_service: Optional[RAGService] = None


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
    vehicle_info: Dict[str, Any],
    dtc_codes: List[str],
    symptoms: str,
) -> DiagnosisResult:
    """
    Convenience function to perform vehicle diagnosis.

    Args:
        vehicle_info: Dictionary with vehicle information
        dtc_codes: List of DTC codes
        symptoms: Free-text description of symptoms

    Returns:
        DiagnosisResult with complete diagnosis information
    """
    service = get_rag_service()
    return await service.diagnose(vehicle_info, dtc_codes, symptoms)


async def get_context(
    query: str,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Convenience function to retrieve context from vector store.

    Args:
        query: Search query text
        top_k: Number of results to return

    Returns:
        List of matching documents with scores
    """
    service = get_rag_service()
    return await service.get_context(query, top_k)


async def generate_response(
    context: Dict[str, Any],
    query: str,
) -> str:
    """
    Convenience function to generate response with context.

    Args:
        context: Context dictionary
        query: User query

    Returns:
        Generated response text
    """
    service = get_rag_service()

    # Convert dict to RAGContext if needed
    if isinstance(context, dict):
        rag_context = RAGContext(
            dtc_context=context.get("dtc_context", []),
            symptom_context=context.get("symptom_context", []),
            graph_context=context.get("graph_context", {}),
        )
    else:
        rag_context = context

    return await service.generate_response(rag_context, query)
