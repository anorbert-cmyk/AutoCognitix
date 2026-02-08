"""
Services module for AutoCognitix.

This module contains service classes and functions for various
application functionalities including NLP, embeddings, LLM providers,
RAG pipeline, and external APIs.
"""

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
from app.services.embedding_service import (
    HungarianEmbeddingService,
    embed_batch,
    embed_text,
    get_embedding_service,
    get_similar_texts,
    preprocess_hungarian,
)
from app.services.llm_provider import (
    AnthropicProvider,
    BaseLLMProvider,
    LLMConfig,
    LLMMessage,
    LLMProviderFactory,
    LLMProviderType,
    LLMResponse,
    LLMStreamChunk,
    OllamaProvider,
    OpenAIProvider,
    RuleBasedProvider,
    get_current_provider_info,
    get_llm_provider,
    is_llm_available,
)
from app.services.llm_provider import (
    generate_response as llm_generate_response,
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
from app.services.rag_service import (
    ConfidenceLevel,
    DiagnosisResult,
    HybridRanker,
    RAGContext,
    RAGService,
    RepairRecommendation,
    RetrievalSource,
    RetrievedItem,
    VehicleInfo,
    diagnose,
    get_context,
    get_rag_service,
)

__all__ = [
    "AnthropicProvider",
    "BaseLLMProvider",
    "Complaint",
    "ConfidenceLevel",
    "DTCValidationError",
    "DiagnosisResult",
    # Diagnosis Service
    "DiagnosisService",
    "DiagnosisServiceError",
    # Embedding Service
    "HungarianEmbeddingService",
    "HybridRanker",
    "LLMConfig",
    "LLMMessage",
    "LLMProviderFactory",
    # LLM Provider
    "LLMProviderType",
    "LLMResponse",
    "LLMStreamChunk",
    "NHTSAError",
    # NHTSA Service
    "NHTSAService",
    "OllamaProvider",
    "OpenAIProvider",
    "RAGContext",
    # RAG Service
    "RAGService",
    "RateLimitError",
    "Recall",
    "RepairRecommendation",
    "RetrievalSource",
    "RetrievedItem",
    "RuleBasedProvider",
    "VINDecodeError",
    "VINDecodeResult",
    "VehicleInfo",
    "analyze_vehicle",
    "close_nhtsa_service",
    "diagnose",
    "embed_batch",
    "embed_text",
    "get_context",
    "get_current_provider_info",
    "get_diagnosis_by_id",
    "get_diagnosis_service",
    "get_embedding_service",
    "get_llm_provider",
    "get_nhtsa_service",
    "get_rag_service",
    "get_similar_texts",
    "get_user_history",
    "is_llm_available",
    "llm_generate_response",
    "preprocess_hungarian",
]
