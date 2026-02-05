"""
Test suite for RAG Pipeline.

Tests the complete RAG pipeline including:
- Prompt template formatting
- Response parsing
- LLM provider abstraction
- Hybrid ranking
- Confidence scoring
- Rule-based fallback

Run with: pytest tests/test_rag_pipeline.py -v
"""

import asyncio
import pytest
from dataclasses import asdict
from typing import Dict, Any, List

# Test imports
from app.prompts.diagnosis_hu import (
    SYSTEM_PROMPT_HU,
    DIAGNOSIS_USER_PROMPT_HU,
    DiagnosisPromptContext,
    build_diagnosis_prompt,
    parse_diagnosis_response,
    ParsedDiagnosisResponse,
    format_dtc_context,
    format_symptom_context,
    format_repair_context,
    format_recall_context,
    generate_rule_based_diagnosis,
)
from app.services.llm_provider import (
    LLMProviderType,
    LLMConfig,
    LLMMessage,
    LLMResponse,
    RuleBasedProvider,
    LLMProviderFactory,
    is_llm_available,
)
from app.services.rag_service import (
    RAGService,
    RAGContext,
    VehicleInfo,
    RetrievedItem,
    RetrievalSource,
    HybridRanker,
    ConfidenceLevel,
)


# =============================================================================
# Test Data
# =============================================================================

SAMPLE_DTC_DATA = [
    {
        "code": "P0171",
        "description_hu": "Rendszer tul szegeny (1. sor)",
        "description_en": "System Too Lean (Bank 1)",
        "category": "powertrain",
        "severity": "medium",
        "symptoms": ["Egyenetlen jaratas", "Megnott fogyasztas"],
        "possible_causes": ["Levegoszivatas", "Hibas MAF szenzor", "Injektorok eldugulaasa"],
        "diagnostic_steps": ["Ellenorizze a levego bevezetot", "Tisztitsa meg a MAF szenzort"],
    },
    {
        "code": "P0300",
        "description_hu": "Veletlenszeru/Tobbszoros cilinder gyujtaskimaradas",
        "description_en": "Random/Multiple Cylinder Misfire Detected",
        "category": "powertrain",
        "severity": "high",
        "symptoms": ["Motor remeg", "Teljesitmeny csokkenes"],
        "possible_causes": ["Elkopott gyujtagyertyak", "Hibas gyujtokabelek"],
    },
]

SAMPLE_VEHICLE_INFO = {
    "make": "Volkswagen",
    "model": "Golf",
    "year": 2018,
    "engine_code": "2.0 TSI",
    "mileage_km": 85000,
    "vin": "WVWZZZ3CZWE123456",
}

SAMPLE_SYMPTOMS = "A motor nehezen indul hidegben, egyenetlenul jar alapjaraton, es a fogyasztas megnott."


# =============================================================================
# Prompt Template Tests
# =============================================================================

class TestPromptTemplates:
    """Tests for Hungarian prompt templates."""

    def test_system_prompt_exists(self):
        """Test that system prompt is defined and not empty."""
        assert SYSTEM_PROMPT_HU is not None
        assert len(SYSTEM_PROMPT_HU) > 100
        assert "diagnosztikai" in SYSTEM_PROMPT_HU.lower()

    def test_format_dtc_context(self):
        """Test DTC context formatting."""
        formatted = format_dtc_context(SAMPLE_DTC_DATA)

        assert "P0171" in formatted
        assert "P0300" in formatted
        assert "Kategoria:" in formatted
        assert "Sulyossag:" in formatted

    def test_format_dtc_context_empty(self):
        """Test DTC context formatting with empty data."""
        formatted = format_dtc_context([])
        assert "Nincs talalat" in formatted

    def test_format_symptom_context(self):
        """Test symptom context formatting."""
        symptom_data = [
            {
                "description": "Motor remeg alapjaraton",
                "score": 0.85,
                "resolution": "Gyujtagyertya csere",
                "related_dtc": ["P0300", "P0301"],
            }
        ]
        formatted = format_symptom_context(symptom_data)

        assert "Motor remeg" in formatted
        assert "85%" in formatted
        assert "P0300" in formatted

    def test_format_repair_context(self):
        """Test repair context formatting."""
        repair_data = {
            "components": [
                {"name": "MAF szenzor", "name_hu": "Levegomero", "system": "Motervezerles"},
            ],
            "repairs": [
                {
                    "name": "MAF szenzor tisztitas",
                    "difficulty": "beginner",
                    "estimated_time_minutes": 30,
                    "estimated_cost_min": 2000,
                    "estimated_cost_max": 5000,
                    "parts": [{"name": "MAF tisztito spray"}],
                }
            ],
            "symptoms": [
                {"name": "Egyenetlen jaratas", "confidence": 0.8},
            ],
        }
        formatted = format_repair_context(repair_data)

        assert "Erintett komponensek" in formatted
        assert "MAF szenzor" in formatted or "Levegomero" in formatted
        assert "beginner" in formatted

    def test_format_recall_context(self):
        """Test recall context formatting."""
        recalls = [
            {
                "campaign_number": "21V123",
                "component": "Futomuro",
                "summary": "A kerekagy csapagyai hibasan lettek osszeszerelve",
                "consequence": "A kerek blokkolhat menet kozben",
            }
        ]
        complaints = [
            {
                "components": "Motor",
                "summary": "A motor leallt menet kozben",
                "crash": True,
                "fire": False,
            }
        ]
        formatted = format_recall_context(recalls, complaints)

        assert "visszahivas" in formatted.lower()
        assert "21V123" in formatted
        assert "BALESET" in formatted

    def test_build_diagnosis_prompt(self):
        """Test complete prompt building."""
        context = DiagnosisPromptContext(
            make="Volkswagen",
            model="Golf",
            year=2018,
            dtc_codes=["P0171", "P0300"],
            symptoms=SAMPLE_SYMPTOMS,
            dtc_context="Test DTC context",
            symptom_context="Test symptom context",
        )

        prompt = build_diagnosis_prompt(context)

        assert "Volkswagen" in prompt
        assert "Golf" in prompt
        assert "2018" in prompt
        assert "P0171" in prompt
        assert "P0300" in prompt
        assert SAMPLE_SYMPTOMS in prompt


# =============================================================================
# Response Parsing Tests
# =============================================================================

class TestResponseParsing:
    """Tests for LLM response parsing."""

    def test_parse_valid_json_response(self):
        """Test parsing valid JSON response."""
        response_text = """
        ```json
        {
            "summary": "A motor szegeny kevereket jelez",
            "probable_causes": [
                {
                    "title": "MAF szenzor hiba",
                    "description": "A levegomero szenzor hibas ertekeket ad",
                    "confidence": 0.85,
                    "related_dtc_codes": ["P0171"],
                    "components": ["MAF szenzor"]
                }
            ],
            "diagnostic_steps": [
                "1. Ellenorizze a MAF szenzort",
                "2. Tisztitsa meg a szenzort"
            ],
            "recommended_repairs": [
                {
                    "title": "MAF szenzor tisztitas",
                    "description": "Tisztitsa meg a szenzort specialan tisztitoszerrel",
                    "estimated_cost_min": 2000,
                    "estimated_cost_max": 5000,
                    "difficulty": "beginner",
                    "parts_needed": ["MAF tisztito"],
                    "estimated_time_minutes": 30
                }
            ],
            "safety_warnings": [],
            "additional_notes": "A problema valoszinuleg a MAF szenzorral kapcsolatos"
        }
        ```
        """

        result = parse_diagnosis_response(response_text)

        assert result.summary == "A motor szegeny kevereket jelez"
        assert len(result.probable_causes) == 1
        assert result.probable_causes[0]["title"] == "MAF szenzor hiba"
        assert result.probable_causes[0]["confidence"] == 0.85
        assert len(result.diagnostic_steps) == 2
        assert len(result.recommended_repairs) == 1
        assert result.parse_error is None

    def test_parse_raw_json_response(self):
        """Test parsing raw JSON without markdown."""
        response_text = """{
            "summary": "Test summary",
            "probable_causes": [],
            "diagnostic_steps": [],
            "recommended_repairs": [],
            "safety_warnings": []
        }"""

        result = parse_diagnosis_response(response_text)

        assert result.summary == "Test summary"
        assert result.parse_error is None

    def test_parse_invalid_response(self):
        """Test parsing invalid response falls back gracefully."""
        response_text = "This is not JSON at all. Just plain text."

        result = parse_diagnosis_response(response_text)

        assert result.parse_error is not None
        assert "No JSON found" in result.parse_error


# =============================================================================
# LLM Provider Tests
# =============================================================================

class TestLLMProvider:
    """Tests for LLM provider abstraction."""

    def test_rule_based_provider_available(self):
        """Test that rule-based provider is always available."""
        provider = RuleBasedProvider()
        assert provider.is_available() == True
        assert provider.provider_type == LLMProviderType.RULE_BASED
        assert provider.model_name == "rule-based-v1"

    @pytest.mark.asyncio
    async def test_rule_based_provider_generate(self):
        """Test rule-based provider generates response."""
        provider = RuleBasedProvider()
        messages = [
            LLMMessage(role="system", content="System prompt"),
            LLMMessage(role="user", content="User prompt"),
        ]

        response = await provider.generate(messages)

        assert response.content is not None
        assert response.provider == LLMProviderType.RULE_BASED
        assert "szabaly-alapu" in response.content.lower()

    def test_factory_auto_detect_returns_provider(self):
        """Test factory returns some provider."""
        provider = LLMProviderFactory.get_provider()
        assert provider is not None
        assert provider.provider_type is not None

    def test_factory_get_available_providers(self):
        """Test getting available providers list."""
        available = LLMProviderFactory.get_available_providers()
        assert LLMProviderType.RULE_BASED in available


# =============================================================================
# Hybrid Ranking Tests
# =============================================================================

class TestHybridRanking:
    """Tests for hybrid ranking (RRF)."""

    def test_reciprocal_rank_fusion_single_list(self):
        """Test RRF with single list returns same items."""
        ranker = HybridRanker()

        items = [
            RetrievedItem(content={"id": 1}, source=RetrievalSource.QDRANT_DTC, score=0.9),
            RetrievedItem(content={"id": 2}, source=RetrievalSource.QDRANT_DTC, score=0.8),
        ]

        result = ranker.reciprocal_rank_fusion([items])

        assert len(result) == 2
        assert result[0].content["id"] == 1
        assert result[1].content["id"] == 2

    def test_reciprocal_rank_fusion_multiple_lists(self):
        """Test RRF with multiple lists combines correctly."""
        ranker = HybridRanker(k=60)

        list1 = [
            RetrievedItem(content={"id": "a"}, source=RetrievalSource.QDRANT_DTC, score=0.9),
            RetrievedItem(content={"id": "b"}, source=RetrievalSource.QDRANT_DTC, score=0.8),
        ]
        list2 = [
            RetrievedItem(content={"id": "b"}, source=RetrievalSource.POSTGRES_TEXT, score=0.95),
            RetrievedItem(content={"id": "c"}, source=RetrievalSource.POSTGRES_TEXT, score=0.7),
        ]

        result = ranker.reciprocal_rank_fusion([list1, list2])

        # Item "b" should be ranked higher because it appears in both lists
        assert len(result) == 3
        # Find item b and verify it has higher combined score
        item_b = next(item for item in result if item.content["id"] == "b")
        item_a = next(item for item in result if item.content["id"] == "a")
        assert item_b.score > item_a.score  # b appears in both lists

    def test_reciprocal_rank_fusion_empty_lists(self):
        """Test RRF with empty lists."""
        ranker = HybridRanker()
        result = ranker.reciprocal_rank_fusion([])
        assert result == []


# =============================================================================
# Rule-Based Diagnosis Tests
# =============================================================================

class TestRuleBasedDiagnosis:
    """Tests for rule-based diagnosis fallback."""

    def test_generate_rule_based_diagnosis(self):
        """Test rule-based diagnosis generation."""
        result = generate_rule_based_diagnosis(
            dtc_codes=SAMPLE_DTC_DATA,
            vehicle_info=SAMPLE_VEHICLE_INFO,
            recalls=[],
            complaints=[],
        )

        assert isinstance(result, ParsedDiagnosisResponse)
        assert result.summary is not None
        assert len(result.summary) > 0
        assert len(result.probable_causes) > 0
        assert len(result.diagnostic_steps) > 0
        assert result.confidence_score > 0

    def test_rule_based_diagnosis_with_recalls(self):
        """Test rule-based diagnosis includes recall warnings."""
        recalls = [
            {
                "component": "Futomuro",
                "consequence": "A kerek blokkolhat menet kozben",
            }
        ]

        result = generate_rule_based_diagnosis(
            dtc_codes=SAMPLE_DTC_DATA,
            vehicle_info=SAMPLE_VEHICLE_INFO,
            recalls=recalls,
        )

        assert any("VISSZAHIVAS" in w for w in result.safety_warnings)

    def test_rule_based_diagnosis_high_severity(self):
        """Test rule-based diagnosis handles high severity codes."""
        high_severity_dtc = [
            {
                "code": "P0300",
                "description_hu": "Gyujtaskimaradas",
                "severity": "critical",
                "possible_causes": ["Motor hiba"],
            }
        ]

        result = generate_rule_based_diagnosis(
            dtc_codes=high_severity_dtc,
            vehicle_info=SAMPLE_VEHICLE_INFO,
        )

        # Should have safety warning for critical severity
        assert len(result.safety_warnings) > 0
        assert any("Kritikus" in w or "azonnal" in w.lower() for w in result.safety_warnings)


# =============================================================================
# Confidence Scoring Tests
# =============================================================================

class TestConfidenceScoring:
    """Tests for confidence calculation."""

    def test_confidence_with_good_context(self):
        """Test confidence calculation with rich context."""
        service = RAGService()

        context = RAGContext(
            dtc_items=[
                RetrievedItem(content={"code": "P0171"}, source=RetrievalSource.QDRANT_DTC, score=0.9),
            ],
            symptom_items=[
                RetrievedItem(content={"description": "Test"}, source=RetrievalSource.QDRANT_SYMPTOM, score=0.85),
            ],
            text_items=[
                RetrievedItem(content={"code": "P0171"}, source=RetrievalSource.POSTGRES_TEXT, score=1.0),
            ],
            graph_data={
                "components": [{"name": "MAF"}],
                "repairs": [{"name": "Clean MAF"}],
                "symptoms": [{"name": "Rough idle"}],
            },
            recall_items=[
                RetrievedItem(content={"component": "Test"}, source=RetrievalSource.NHTSA, score=0.9,
                             metadata={"type": "recall"}),
            ],
        )

        level, score = service.calculate_confidence(context, ["P0171"])

        assert score > 0.5
        assert level in [ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM]

    def test_confidence_with_empty_context(self):
        """Test confidence calculation with empty context."""
        service = RAGService()
        context = RAGContext()

        level, score = service.calculate_confidence(context, ["P0171"])

        assert score < 0.3
        assert level in [ConfidenceLevel.LOW, ConfidenceLevel.UNKNOWN]


# =============================================================================
# Integration Test (requires running services)
# =============================================================================

@pytest.mark.asyncio
@pytest.mark.integration
async def test_full_rag_pipeline():
    """
    Integration test for full RAG pipeline.

    Note: This test requires:
    - Qdrant running and initialized
    - Neo4j running with seed data
    - PostgreSQL running with DTC codes

    Run with: pytest tests/test_rag_pipeline.py -v -m integration
    """
    from app.services.rag_service import diagnose

    vehicle_info = SAMPLE_VEHICLE_INFO
    dtc_codes = ["P0171"]
    symptoms = SAMPLE_SYMPTOMS

    # This will use rule-based fallback if no LLM is configured
    result = await diagnose(
        vehicle_info=vehicle_info,
        dtc_codes=dtc_codes,
        symptoms=symptoms,
    )

    assert result is not None
    assert result.dtc_codes == ["P0171"]
    assert result.diagnosis_summary is not None
    assert result.confidence_score >= 0
    assert result.confidence_score <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
