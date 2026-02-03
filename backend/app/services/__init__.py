"""
Services module for AutoCognitix.

This module contains service classes and functions for various
application functionalities including NLP, embeddings, and external APIs.
"""

from app.services.embedding_service import (
    HungarianEmbeddingService,
    embed_batch,
    embed_text,
    get_embedding_service,
    get_similar_texts,
    preprocess_hungarian,
)
from app.services.nhtsa_service import (
    Complaint,
    NHTSAError,
    NHTSAService,
    RateLimitError,
    Recall,
    VINDecodeResult,
    close_nhtsa_service,
    get_nhtsa_service,
)
from app.services.diagnosis_service import (
    DiagnosisService,
    DiagnosisServiceError,
    DTCValidationError,
    VINDecodeError,
    analyze_vehicle,
    get_diagnosis_by_id,
    get_diagnosis_service,
    get_user_history,
)
from app.services.rag_service import (
    RAGService,
    RAGContext,
    DiagnosisResult,
    VehicleInfo,
    RepairRecommendation,
    ConfidenceLevel,
    LLMProvider,
    get_rag_service,
    diagnose,
    get_context,
    generate_response,
)

__all__ = [
    # Embedding Service
    "HungarianEmbeddingService",
    "get_embedding_service",
    "embed_text",
    "embed_batch",
    "preprocess_hungarian",
    "get_similar_texts",
    # NHTSA Service
    "NHTSAService",
    "get_nhtsa_service",
    "close_nhtsa_service",
    "VINDecodeResult",
    "Recall",
    "Complaint",
    "NHTSAError",
    "RateLimitError",
    # Diagnosis Service
    "DiagnosisService",
    "DiagnosisServiceError",
    "DTCValidationError",
    "VINDecodeError",
    "analyze_vehicle",
    "get_diagnosis_by_id",
    "get_diagnosis_service",
    "get_user_history",
    # RAG Service
    "RAGService",
    "RAGContext",
    "DiagnosisResult",
    "VehicleInfo",
    "RepairRecommendation",
    "ConfidenceLevel",
    "LLMProvider",
    "get_rag_service",
    "diagnose",
    "get_context",
    "generate_response",
]
