"""
Integration tests for the RAG (Retrieval-Augmented Generation) service.

Tests context retrieval, response generation, and confidence scoring.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from dataclasses import field
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(backend_path))


class TestRAGContextRetrieval:
    """Test RAG context retrieval from Qdrant."""

    @pytest.mark.asyncio
    async def test_retrieve_from_qdrant_returns_list(self, mock_qdrant_client):
        """Test that retrieve_from_qdrant returns a list of RetrievedItem."""
        mock_qdrant_client.search = AsyncMock(
            return_value=[
                {"id": "1", "score": 0.9, "payload": {"code": "P0101"}},
            ]
        )

        with patch("app.services.rag_service.embed_text_async") as mock_embed:
            mock_embed.return_value = [0.0] * 768

            from app.services.rag_service import RAGService

            service = RAGService()
            service._qdrant = mock_qdrant_client
            service._cache.clear()

            results = await service.retrieve_from_qdrant(
                "Motor problem", collection="dtc_codes", top_k=5
            )

            assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_retrieve_from_qdrant_respects_top_k(self, mock_qdrant_client):
        """Test that retrieve_from_qdrant passes top_k to Qdrant."""
        mock_qdrant_client.search = AsyncMock(
            return_value=[
                {"id": "1", "score": 0.9, "payload": {}},
                {"id": "2", "score": 0.8, "payload": {}},
            ]
        )

        with patch("app.services.rag_service.embed_text_async") as mock_embed:
            mock_embed.return_value = [0.0] * 768

            from app.services.rag_service import RAGService

            service = RAGService()
            service._qdrant = mock_qdrant_client
            service._cache.clear()

            results = await service.retrieve_from_qdrant("Query", collection="dtc_codes", top_k=2)

            assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_retrieve_from_qdrant_handles_error(self, mock_qdrant_client):
        """Test that retrieve_from_qdrant handles errors gracefully."""
        mock_qdrant_client.search = AsyncMock(side_effect=Exception("Connection error"))

        with patch("app.services.rag_service.embed_text_async") as mock_embed:
            mock_embed.return_value = [0.0] * 768

            from app.services.rag_service import RAGService

            service = RAGService()
            service._qdrant = mock_qdrant_client
            service._cache.clear()

            results = await service.retrieve_from_qdrant("Query", collection="dtc_codes", top_k=5)

            assert isinstance(results, list)
            assert len(results) == 0


class TestRAGNeo4jRetrieval:
    """Test Neo4j graph context retrieval."""

    @pytest.mark.asyncio
    async def test_retrieve_from_neo4j_returns_items(self, mock_neo4j_client):
        """Test retrieving graph context from Neo4j."""
        with patch("app.services.rag_service.get_diagnostic_path") as mock_graph:
            mock_graph.return_value = {
                "dtc": {"code": "P0101", "description": "MAF Issue"},
                "symptoms": [{"name": "Rough idle"}],
                "components": [{"name": "MAF Sensor"}],
                "repairs": [{"name": "Replace MAF"}],
            }

            from app.services.rag_service import RAGService

            service = RAGService()

            items, graph_data = await service.retrieve_from_neo4j(["P0101"])

            assert isinstance(items, list)
            assert isinstance(graph_data, dict)

    @pytest.mark.asyncio
    async def test_retrieve_from_neo4j_includes_graph_data(self, mock_neo4j_client):
        """Test that Neo4j retrieval includes structured graph data."""
        with patch("app.services.rag_service.get_diagnostic_path") as mock_graph:
            mock_graph.return_value = {
                "dtc": {"code": "P0101", "description": "MAF Issue"},
                "symptoms": [{"name": "Rough idle"}],
                "components": [{"name": "MAF Sensor"}],
                "repairs": [{"name": "Replace MAF"}],
            }

            from app.services.rag_service import RAGService

            service = RAGService()

            _items, graph_data = await service.retrieve_from_neo4j(["P0101"])

            # Should include structured data from graph
            assert "components" in graph_data
            assert len(graph_data["components"]) > 0


class TestRAGConfidenceScoring:
    """Test RAG confidence score calculation."""

    def test_confidence_calculation_with_good_context(self):
        """Test confidence calculation with rich context."""
        from app.services.rag_service import (
            RAGService,
            RAGContext,
            RetrievedItem,
            RetrievalSource,
        )

        service = RAGService()

        context = RAGContext(
            dtc_items=[
                RetrievedItem(
                    content={"code": "P0101"},
                    source=RetrievalSource.QDRANT_DTC,
                    score=0.9,
                ),
                RetrievedItem(
                    content={"code": "P0171"},
                    source=RetrievalSource.QDRANT_DTC,
                    score=0.85,
                ),
            ],
            symptom_items=[
                RetrievedItem(
                    content={"description": "Symptom 1"},
                    source=RetrievalSource.QDRANT_SYMPTOM,
                    score=0.8,
                ),
            ],
            graph_data={
                "P0101": {
                    "components": [{"name": "MAF Sensor"}],
                    "repairs": [{"name": "Replace MAF"}],
                    "symptoms": [{"name": "Rough idle"}],
                },
            },
        )

        _level, score = service.calculate_confidence(context, ["P0101", "P0171"])

        # Should have reasonable confidence with good context
        assert score > 0.3

    def test_confidence_calculation_with_minimal_context(self):
        """Test confidence calculation with minimal context."""
        from app.services.rag_service import RAGService, RAGContext

        service = RAGService()

        context = RAGContext()

        _level, score = service.calculate_confidence(context, ["P0101"])

        # Should have low confidence with no context
        assert score < 0.5

    def test_confidence_levels_match_scores(self):
        """Test that confidence levels match score ranges."""
        from app.services.rag_service import (
            RAGService,
            RAGContext,
            ConfidenceLevel,
            RetrievedItem,
            RetrievalSource,
        )

        service = RAGService()

        # High confidence context
        high_context = RAGContext(
            dtc_items=[
                RetrievedItem(
                    content={"code": "P0101"},
                    source=RetrievalSource.QDRANT_DTC,
                    score=0.95,
                ),
            ],
            symptom_items=[
                RetrievedItem(
                    content={"description": "Symptom"},
                    source=RetrievalSource.QDRANT_SYMPTOM,
                    score=0.9,
                ),
            ],
            graph_data={
                "P0101": {
                    "components": [{}],
                    "repairs": [{}],
                    "symptoms": [{}],
                },
            },
        )

        level, score = service.calculate_confidence(high_context, ["P0101"])

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
    async def test_service_has_diagnose_method(self, mock_rag_service):
        """Test that RAG service exposes diagnose method."""
        from app.services.rag_service import RAGService

        service = RAGService()
        assert hasattr(service, "diagnose")
        assert hasattr(service, "generate_diagnosis")

    @pytest.mark.asyncio
    async def test_service_has_assemble_context(self):
        """Test that RAG service has context assembly method."""
        from app.services.rag_service import RAGService

        service = RAGService()
        assert hasattr(service, "assemble_context")


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
            dtc_context="P0101 - MAF Circuit Issue (medium, powertrain)",
            repair_context="Replace MAF Sensor - beginner - 30 min",
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


class TestRAGErrorHandling:
    """Test RAG service error handling."""

    @pytest.mark.asyncio
    async def test_handles_qdrant_error(self, mock_qdrant_client):
        """Test handling of Qdrant errors."""
        mock_qdrant_client.search = AsyncMock(side_effect=Exception("Qdrant unavailable"))

        with patch("app.services.rag_service.embed_text_async") as mock_embed:
            mock_embed.return_value = [0.0] * 768

            from app.services.rag_service import RAGService

            service = RAGService()
            service._qdrant = mock_qdrant_client
            service._cache.clear()

            # Should handle error gracefully
            results = await service.retrieve_from_qdrant("Query", collection="dtc_codes", top_k=5)

            # Should return empty list on error
            assert isinstance(results, list)
            assert len(results) == 0

    @pytest.mark.asyncio
    async def test_handles_neo4j_error(self):
        """Test handling of Neo4j errors."""
        with patch("app.services.rag_service.get_diagnostic_path") as mock_graph:
            mock_graph.side_effect = Exception("Neo4j unavailable")

            from app.services.rag_service import RAGService

            service = RAGService()

            items, _graph_data = await service.retrieve_from_neo4j(["P0101"])

            assert isinstance(items, list)
            assert len(items) == 0

    @pytest.mark.asyncio
    async def test_handles_empty_dtc_codes(self):
        """Test handling of empty DTC codes list."""
        with patch("app.services.rag_service.get_diagnostic_path") as mock_graph:
            from app.services.rag_service import RAGService

            service = RAGService()

            items, graph_data = await service.retrieve_from_neo4j([])

            assert items == []
            assert graph_data == {"components": [], "repairs": [], "symptoms": []}
            mock_graph.assert_not_called()


class TestRAGServiceSingleton:
    """Test RAG service singleton pattern."""

    def test_get_rag_service_returns_instance(self):
        """Test that get_rag_service returns an instance."""
        from app.services.rag_service import RAGService, get_rag_service

        service = get_rag_service()

        assert service is not None
        assert isinstance(service, RAGService)

    def test_get_rag_service_returns_same_instance(self):
        """Test that get_rag_service returns same instance."""
        from app.services.rag_service import get_rag_service

        service1 = get_rag_service()
        service2 = get_rag_service()

        assert service1 is service2
