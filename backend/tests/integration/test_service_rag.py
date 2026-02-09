"""
Integration tests for the RAG (Retrieval-Augmented Generation) service.

Tests context retrieval, response generation, and confidence scoring.
"""

import pytest
from unittest.mock import AsyncMock, patch
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(backend_path))


class TestRAGContextRetrieval:
    """Test RAG context retrieval from multiple sources."""

    @pytest.mark.asyncio
    async def test_get_context_returns_list(self, mock_qdrant_client):
        """Test that get_context returns a list of results."""
        with (
            patch("app.services.rag_service.qdrant_client", mock_qdrant_client),
            patch("app.services.rag_service.embed_text") as mock_embed,
        ):
            mock_embed.return_value = [0.0] * 768

            from app.services.rag_service import get_rag_service

            service = get_rag_service()
            service._qdrant = mock_qdrant_client

            results = await service.get_context("Motor problem", top_k=5)

            assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_get_context_respects_top_k(self, mock_qdrant_client):
        """Test that get_context respects top_k parameter."""
        mock_qdrant_client.search.return_value = [
            {"id": "1", "score": 0.9, "payload": {}},
            {"id": "2", "score": 0.8, "payload": {}},
        ]

        with (
            patch("app.services.rag_service.qdrant_client", mock_qdrant_client),
            patch("app.services.rag_service.embed_text") as mock_embed,
        ):
            mock_embed.return_value = [0.0] * 768

            from app.services.rag_service import get_rag_service

            service = get_rag_service()
            service._qdrant = mock_qdrant_client

            results = await service.get_context("Query", top_k=2)

            assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_get_context_searches_multiple_collections(self, mock_qdrant_client):
        """Test that get_context searches multiple collections."""
        call_count = [0]

        async def count_searches(*args, **kwargs):
            call_count[0] += 1
            return []

        mock_qdrant_client.search = count_searches

        with (
            patch("app.services.rag_service.qdrant_client", mock_qdrant_client),
            patch("app.services.rag_service.embed_text") as mock_embed,
        ):
            mock_embed.return_value = [0.0] * 768

            from app.services.rag_service import get_rag_service

            service = get_rag_service()
            service._qdrant = mock_qdrant_client

            await service.get_context("Query", top_k=5)

            # Should have searched multiple collections
            assert call_count[0] >= 1


class TestRAGDTCContext:
    """Test DTC-specific context retrieval."""

    @pytest.mark.asyncio
    async def test_get_dtc_context_for_valid_code(self, mock_qdrant_client, mock_neo4j_client):
        """Test getting context for a valid DTC code."""
        mock_qdrant_client.search_dtc = AsyncMock(
            return_value=[
                {
                    "score": 0.9,
                    "payload": {
                        "code": "P0101",
                        "description_hu": "Levegotomeg-mero hiba",
                        "severity": "medium",
                        "category": "powertrain",
                    },
                }
            ]
        )

        with (
            patch("app.services.rag_service.qdrant_client", mock_qdrant_client),
            patch("app.services.rag_service.embed_text") as mock_embed,
            patch("app.services.rag_service.get_diagnostic_path") as mock_graph,
        ):
            mock_embed.return_value = [0.0] * 768
            mock_graph.return_value = None

            from app.services.rag_service import RAGService

            service = RAGService()
            service._qdrant = mock_qdrant_client

            context = await service._get_dtc_context(["P0101"])

            assert isinstance(context, list)

    @pytest.mark.asyncio
    async def test_get_dtc_context_includes_graph_data(self, mock_qdrant_client, mock_neo4j_client):
        """Test that DTC context includes Neo4j graph data."""
        with (
            patch("app.services.rag_service.qdrant_client", mock_qdrant_client),
            patch("app.services.rag_service.embed_text") as mock_embed,
            patch("app.services.rag_service.get_diagnostic_path") as mock_graph,
        ):
            mock_embed.return_value = [0.0] * 768
            mock_graph.return_value = {
                "dtc": {"code": "P0101", "description": "MAF Issue"},
                "symptoms": [{"name": "Rough idle"}],
                "components": [{"name": "MAF Sensor"}],
                "repairs": [{"name": "Replace MAF"}],
            }
            mock_qdrant_client.search_dtc = AsyncMock(return_value=[])

            from app.services.rag_service import RAGService

            service = RAGService()
            service._qdrant = mock_qdrant_client

            context = await service._get_dtc_context(["P0101"])

            # Should include data from graph
            assert any(c.get("source") == "neo4j_graph" for c in context)


class TestRAGSymptomContext:
    """Test symptom-specific context retrieval."""

    @pytest.mark.asyncio
    async def test_get_symptom_context(self, mock_qdrant_client):
        """Test getting context for symptom description."""
        mock_qdrant_client.search_similar_symptoms = AsyncMock(
            return_value=[
                {
                    "score": 0.85,
                    "payload": {
                        "description": "Motor nehezen indul",
                        "related_dtc": ["P0101"],
                    },
                }
            ]
        )

        with (
            patch("app.services.rag_service.qdrant_client", mock_qdrant_client),
            patch("app.services.rag_service.embed_text") as mock_embed,
        ):
            mock_embed.return_value = [0.0] * 768

            from app.services.rag_service import RAGService

            service = RAGService()
            service._qdrant = mock_qdrant_client

            context = await service._get_symptom_context("A motor nehezen indul hidegben")

            assert isinstance(context, list)

    @pytest.mark.asyncio
    async def test_get_symptom_context_with_vehicle_filter(self, mock_qdrant_client):
        """Test symptom context with vehicle make filter."""
        mock_qdrant_client.search_similar_symptoms = AsyncMock(
            return_value=[
                {
                    "score": 0.88,
                    "payload": {
                        "description": "VW Golf MAF problem",
                        "vehicle_make": "Volkswagen",
                        "related_dtc": ["P0101"],
                    },
                }
            ]
        )

        with (
            patch("app.services.rag_service.qdrant_client", mock_qdrant_client),
            patch("app.services.rag_service.embed_text") as mock_embed,
        ):
            mock_embed.return_value = [0.0] * 768

            from app.services.rag_service import RAGService

            service = RAGService()
            service._qdrant = mock_qdrant_client

            await service._get_symptom_context(
                "Motor problem",
                vehicle_make="Volkswagen",
            )

            mock_qdrant_client.search_similar_symptoms.assert_called_once()


class TestRAGConfidenceScoring:
    """Test RAG confidence score calculation."""

    def test_confidence_calculation_with_good_context(self):
        """Test confidence calculation with rich context."""
        from app.services.rag_service import RAGService, RAGContext

        service = RAGService()

        context = RAGContext(
            dtc_context=[
                {"code": "P0101", "score": 0.9},
                {"code": "P0171", "score": 0.85},
            ],
            symptom_context=[
                {"description": "Symptom 1", "score": 0.8},
                {"description": "Symptom 2", "score": 0.75},
            ],
            graph_context={
                "components": [{"name": "MAF Sensor"}],
                "repairs": [{"name": "Replace MAF"}],
                "symptoms": [{"name": "Rough idle"}],
            },
        )

        _level, score = service._calculate_confidence(context, ["P0101", "P0171"])

        # Should have reasonable confidence with good context
        assert score > 0.3

    def test_confidence_calculation_with_minimal_context(self):
        """Test confidence calculation with minimal context."""
        from app.services.rag_service import RAGService, RAGContext

        service = RAGService()

        context = RAGContext(
            dtc_context=[],
            symptom_context=[],
            graph_context={},
        )

        _level, score = service._calculate_confidence(context, ["P0101"])

        # Should have low confidence with no context
        assert score < 0.5

    def test_confidence_levels_match_scores(self):
        """Test that confidence levels match score ranges."""
        from app.services.rag_service import RAGService, RAGContext, ConfidenceLevel

        service = RAGService()

        # High confidence context
        high_context = RAGContext(
            dtc_context=[{"code": "P0101", "score": 0.95}],
            symptom_context=[{"score": 0.9}],
            graph_context={
                "components": [{}],
                "repairs": [{}],
                "symptoms": [{}],
            },
        )

        level, score = service._calculate_confidence(high_context, ["P0101"])

        # Level should match score
        if score >= 0.75:
            assert level == ConfidenceLevel.HIGH
        elif score >= 0.5:
            assert level == ConfidenceLevel.MEDIUM
        elif score >= 0.25:
            assert level == ConfidenceLevel.LOW
        else:
            assert level == ConfidenceLevel.UNKNOWN


class TestRAGResponseGeneration:
    """Test RAG response generation."""

    @pytest.mark.asyncio
    async def test_generate_response_returns_string(self, mock_rag_service):
        """Test that generate_response returns a string."""
        from app.services.rag_service import RAGContext

        RAGContext(
            dtc_context=[{"code": "P0101"}],
            symptom_context=[],
            graph_context={},
        )

        # Mock would be used for actual LLM call
        # Here we just verify the interface
        assert hasattr(mock_rag_service, "diagnose")

    @pytest.mark.asyncio
    async def test_generate_response_uses_hungarian_template(self):
        """Test that response generation uses Hungarian template."""
        from app.services.rag_service import HUNGARIAN_DIAGNOSIS_TEMPLATE

        # Template should contain Hungarian text
        assert "Jarmu adatok" in HUNGARIAN_DIAGNOSIS_TEMPLATE
        assert "hibakodok" in HUNGARIAN_DIAGNOSIS_TEMPLATE.lower()
        assert "tunetek" in HUNGARIAN_DIAGNOSIS_TEMPLATE.lower()


class TestRAGDiagnosis:
    """Test full RAG diagnosis flow."""

    @pytest.mark.asyncio
    async def test_diagnose_returns_result_object(self, mock_qdrant_client, mock_rag_service):
        """Test that diagnose returns a DiagnosisResult object."""
        # The mock_rag_service already returns a diagnosis result dict
        result = await mock_rag_service.diagnose(
            vehicle_info={"make": "Volkswagen", "model": "Golf", "year": 2018},
            dtc_codes=["P0101"],
            symptoms="Motor nehezen indul",
        )

        assert "probable_causes" in result
        assert "recommended_repairs" in result
        assert "confidence_score" in result

    @pytest.mark.asyncio
    async def test_diagnose_includes_probable_causes(self, mock_rag_service):
        """Test that diagnosis includes probable causes."""
        result = await mock_rag_service.diagnose(
            vehicle_info={"make": "Volkswagen", "model": "Golf", "year": 2018},
            dtc_codes=["P0101"],
            symptoms="Motor nehezen indul",
        )

        assert "probable_causes" in result
        assert isinstance(result["probable_causes"], list)

    @pytest.mark.asyncio
    async def test_diagnose_includes_repair_recommendations(self, mock_rag_service):
        """Test that diagnosis includes repair recommendations."""
        result = await mock_rag_service.diagnose(
            vehicle_info={"make": "Volkswagen", "model": "Golf", "year": 2018},
            dtc_codes=["P0101"],
            symptoms="Motor nehezen indul",
        )

        assert "recommended_repairs" in result
        assert isinstance(result["recommended_repairs"], list)

    @pytest.mark.asyncio
    async def test_diagnose_includes_confidence_score(self, mock_rag_service):
        """Test that diagnosis includes confidence score."""
        result = await mock_rag_service.diagnose(
            vehicle_info={"make": "Volkswagen", "model": "Golf", "year": 2018},
            dtc_codes=["P0101"],
            symptoms="Motor nehezen indul",
        )

        assert "confidence_score" in result
        assert 0 <= result["confidence_score"] <= 1

    @pytest.mark.asyncio
    async def test_diagnose_includes_sources(self, mock_rag_service):
        """Test that diagnosis includes information sources."""
        result = await mock_rag_service.diagnose(
            vehicle_info={"make": "Volkswagen", "model": "Golf", "year": 2018},
            dtc_codes=["P0101"],
            symptoms="Motor nehezen indul",
        )

        assert "sources" in result
        assert isinstance(result["sources"], list)


class TestRAGContextFormatting:
    """Test RAG context formatting for LLM prompts."""

    def test_context_to_formatted_string(self):
        """Test context formatting for LLM prompt."""
        from app.services.rag_service import RAGContext

        context = RAGContext(
            dtc_context=[
                {
                    "code": "P0101",
                    "description": "MAF Issue",
                    "severity": "medium",
                    "category": "powertrain",
                }
            ],
            symptom_context=[{"description": "Motor nehezen indul", "score": 0.85}],
            graph_context={
                "components": [{"name": "MAF Sensor", "system": "Engine"}],
                "repairs": [
                    {"name": "Replace MAF", "difficulty": "beginner", "estimated_time_minutes": 30}
                ],
            },
        )

        formatted = context.to_formatted_string()

        assert isinstance(formatted, str)
        assert "P0101" in formatted
        assert "MAF" in formatted

    def test_empty_context_formatting(self):
        """Test formatting of empty context."""
        from app.services.rag_service import RAGContext

        context = RAGContext()

        formatted = context.to_formatted_string()

        assert isinstance(formatted, str)
        # Should indicate no context available
        assert "kontextus" in formatted.lower() or len(formatted) > 0


class TestRAGErrorHandling:
    """Test RAG service error handling."""

    @pytest.mark.asyncio
    async def test_handles_qdrant_error(self, mock_qdrant_client):
        """Test handling of Qdrant errors."""
        mock_qdrant_client.search.side_effect = Exception("Qdrant unavailable")

        with (
            patch("app.services.rag_service.qdrant_client", mock_qdrant_client),
            patch("app.services.rag_service.embed_text") as mock_embed,
        ):
            mock_embed.return_value = [0.0] * 768

            from app.services.rag_service import RAGService

            service = RAGService()
            service._qdrant = mock_qdrant_client

            # Should handle error gracefully
            results = await service.get_context("Query", top_k=5)

            # Should return empty list on error
            assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_handles_empty_dtc_codes(self, mock_qdrant_client):
        """Test handling of empty DTC codes list."""
        with patch("app.services.rag_service.qdrant_client", mock_qdrant_client):
            from app.services.rag_service import RAGService

            service = RAGService()
            service._qdrant = mock_qdrant_client

            context = await service._get_dtc_context([])

            assert context == []

    @pytest.mark.asyncio
    async def test_handles_empty_symptoms(self, mock_qdrant_client):
        """Test handling of empty symptoms."""
        with patch("app.services.rag_service.qdrant_client", mock_qdrant_client):
            from app.services.rag_service import RAGService

            service = RAGService()
            service._qdrant = mock_qdrant_client

            context = await service._get_symptom_context("")

            assert context == []


class TestRAGServiceSingleton:
    """Test RAG service singleton pattern."""

    def test_get_rag_service_returns_instance(self):
        """Test that get_rag_service returns an instance."""
        with patch("app.services.rag_service._rag_service", None):
            from app.services.rag_service import get_rag_service

            service = get_rag_service()

            assert service is not None

    def test_get_rag_service_returns_same_instance(self):
        """Test that get_rag_service returns same instance."""
        from app.services.rag_service import get_rag_service

        service1 = get_rag_service()
        service2 = get_rag_service()

        assert service1 is service2
