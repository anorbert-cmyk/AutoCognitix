"""
Integration tests for the RAG (Retrieval-Augmented Generation) service.

Tests context retrieval, response generation, and confidence scoring.
"""

import pytest
from functools import partial
from unittest.mock import AsyncMock, MagicMock, patch
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(backend_path))


_LEGACY_COLLECTION_NAMES = (
    "dtc_embeddings_hu",
    "symptom_embeddings_hu",
    "component_embeddings_hu",
    "repair_embeddings_hu",
    "known_issue_embeddings_hu",
)


class TestRAGContextRetrieval:
    """Test RAG context retrieval from Qdrant."""

    @pytest.mark.asyncio
    async def test_retrieve_from_qdrant_returns_list(self, mock_qdrant_client):
        """Test that retrieve_from_qdrant returns a list of RetrievedItem."""
        mock_qdrant_client.search_unified = AsyncMock(
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

            results = await service.retrieve_from_qdrant("Motor problem", type_="dtc", top_k=5)

            assert isinstance(results, list)
            assert results[0].content == {"code": "P0101"}

    @pytest.mark.asyncio
    async def test_retrieve_from_qdrant_respects_top_k(self, mock_qdrant_client):
        """Test that retrieve_from_qdrant passes top_k to Qdrant."""
        mock_qdrant_client.search_unified = AsyncMock(
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

            results = await service.retrieve_from_qdrant("Query", type_="dtc", top_k=2)

            assert len(results) <= 2
            assert mock_qdrant_client.search_unified.await_args.kwargs["limit"] == 2

    @pytest.mark.asyncio
    async def test_retrieve_from_qdrant_handles_error(self, mock_qdrant_client):
        """Test that retrieve_from_qdrant handles errors gracefully."""
        mock_qdrant_client.search_unified = AsyncMock(side_effect=Exception("Connection error"))

        with patch("app.services.rag_service.embed_text_async") as mock_embed:
            mock_embed.return_value = [0.0] * 768

            from app.services.rag_service import RAGService

            service = RAGService()
            service._qdrant = mock_qdrant_client
            service._cache.clear()

            results = await service.retrieve_from_qdrant("Query", type_="dtc", top_k=5)

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
        mock_qdrant_client.search_unified = AsyncMock(side_effect=Exception("Qdrant unavailable"))

        with patch("app.services.rag_service.embed_text_async") as mock_embed:
            mock_embed.return_value = [0.0] * 768

            from app.services.rag_service import RAGService

            service = RAGService()
            service._qdrant = mock_qdrant_client
            service._cache.clear()

            # Should handle error gracefully
            results = await service.retrieve_from_qdrant("Query", type_="dtc", top_k=5)

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


# =============================================================================
# Unified Qdrant collection routing (collection-drift guard)
# =============================================================================

# Unified-collection hits: every huBERT vector lives in ONE collection with a
# type-discriminated payload (the per-type collections were never populated).
UNIFIED_DTC_HITS = [
    {
        "id": 1,
        "score": 0.83,
        "payload": {
            "type": "dtc",
            "code": "P0301",
            "description": "1. henger egeskimaradas",
            "category": "powertrain",
        },
    }
]

UNIFIED_COMPLAINT_HITS = [
    {
        "id": 2,
        "score": 0.71,
        "payload": {
            "type": "complaint",
            "odi_id": "11554321",
            "make": "VOLKSWAGEN",
            "model": "GOLF",
            "year": "2018",
            "component": "ENGINE",
            "description": "Engine misfires and shakes at idle",
        },
    }
]


def _unified_qdrant_mock(hits_by_type=None, error=None):
    """Build a QdrantService double whose ``search_unified`` is the REAL one.

    Only the low-level ``search`` is stubbed, so the assertions observe the
    exact collection name and payload filters that the production code path
    sends to Qdrant - a mocked ``search_unified`` could not prove either.
    """
    from app.db.qdrant_client import QdrantService

    mock = MagicMock()

    if error is not None:
        mock.search = AsyncMock(side_effect=error)
    else:

        async def _search(**kwargs):
            payload_type = (kwargs.get("filter_conditions") or {}).get("type")
            return list((hits_by_type or {}).get(payload_type, []))

        mock.search = AsyncMock(side_effect=_search)

    mock.search_unified = partial(QdrantService.search_unified, mock)
    return mock


@pytest.fixture
def rag_service():
    """RAGService singleton with a clean cache, restored after the test."""
    from app.services.rag_service import RAGService

    service = RAGService()
    original_qdrant = service._qdrant
    service._cache.clear()
    service.set_db_session(None)
    yield service
    service._qdrant = original_qdrant
    service._cache.clear()


class TestRAGUnifiedCollectionRouting:
    """The RAG must query the collection that actually holds the huBERT
    vectors (``settings.QDRANT_UNIFIED_COLLECTION``), not the empty legacy
    per-type collections that the loaders never populated.
    """

    @pytest.mark.asyncio
    async def test_dtc_retrieval_targets_unified_collection_with_type_filter(self, rag_service):
        """A type_="dtc" retrieval hits the unified collection + discriminator."""
        from app.core.config import settings
        from app.services.rag_service import RetrievalSource

        rag_service._qdrant = _unified_qdrant_mock({"dtc": UNIFIED_DTC_HITS})

        with patch(
            "app.services.rag_service.embed_text_async",
            new=AsyncMock(return_value=[0.1] * 768),
        ):
            items = await rag_service.retrieve_from_qdrant(
                query="P0301 egyenetlen jaratas", type_="dtc", top_k=10, preprocess=False
            )

        _, kwargs = rag_service._qdrant.search.call_args
        assert kwargs["filter_conditions"]["type"] == "dtc"
        # The collection is not observable here BECAUSE it is no longer a
        # parameter: QdrantService.search resolves it from settings itself.
        assert "collection_name" not in kwargs
        assert settings.QDRANT_UNIFIED_COLLECTION == "autocognitix"

        assert len(items) == 1
        assert items[0].content["code"] == "P0301"
        assert items[0].source == RetrievalSource.QDRANT_DTC

    @pytest.mark.asyncio
    async def test_revert_guard_rag_never_queries_empty_legacy_collections(self, rag_service):
        """REVERT-GUARD: FAILS if the RAG retrieval legs are pointed back at the
        empty ``dtc_embeddings_hu`` / ``symptom_embeddings_hu`` collections.
        """
        from app.core.config import settings
        from app.db.qdrant_client import QdrantService
        from app.services.rag_service import VehicleInfo

        rag_service._qdrant = _unified_qdrant_mock(
            {"dtc": UNIFIED_DTC_HITS, "complaint": UNIFIED_COMPLAINT_HITS}
        )

        with (
            patch(
                "app.services.rag_service.embed_text_async", new=AsyncMock(return_value=[0.1] * 768)
            ),
            patch("app.services.rag_service.preprocess_hungarian", side_effect=lambda text: text),
            patch("app.services.rag_service.get_diagnostic_path", new=AsyncMock(return_value={})),
        ):
            context = await rag_service.assemble_context(
                vehicle_info=VehicleInfo(make="Volkswagen", model="Golf", year=2018),
                dtc_codes=["P0301"],
                symptoms="egyenetlen jaratas",
            )

        calls = rag_service._qdrant.search.call_args_list
        assert calls, "assemble_context issued no Qdrant search at all"
        # No call may name a collection at all - the parameter is gone, so a
        # retrieval leg cannot be pointed anywhere but settings.QDRANT_UNIFIED_COLLECTION.
        assert all("collection_name" not in call.kwargs for call in calls)
        assert not any(arg in _LEGACY_COLLECTION_NAMES for call in calls for arg in call.args)
        assert {call.kwargs["filter_conditions"]["type"] for call in calls} == {
            "dtc",
            "complaint",
        }

        # ...and the retrieved payloads really reach the prompt context.
        assert "P0301" in context.dtc_context
        assert "Engine misfires" in context.symptom_context

    @pytest.mark.asyncio
    async def test_chat_dtc_context_targets_unified_collection(self, rag_service):
        """REVERT-GUARD for the chat assistant's DTC-context leg.

        ``ChatService._fetch_rag_context`` is a SECOND caller of
        ``retrieve_from_qdrant``; pointing it back at ``dtc_embeddings_hu``
        would silently strip every DTC fact out of the chat prompt without any
        error surfacing. FAILS if the legacy collection reappears.
        """
        from app.core.config import settings
        from app.db.qdrant_client import QdrantService
        from app.services.chat_service import ChatService

        rag_service._qdrant = _unified_qdrant_mock({"dtc": UNIFIED_DTC_HITS})

        with (
            patch(
                "app.services.rag_service.embed_text_async", new=AsyncMock(return_value=[0.1] * 768)
            ),
            patch("app.services.rag_service.get_rag_service", return_value=rag_service),
        ):
            context = await ChatService()._fetch_rag_context(["P0301"])

        calls = rag_service._qdrant.search.call_args_list
        assert calls, "chat RAG context issued no Qdrant search at all"
        assert all("collection_name" not in call.kwargs for call in calls)
        assert all(call.kwargs["filter_conditions"]["type"] == "dtc" for call in calls)

        # ...and the hit really reaches the chat prompt.
        assert context is not None
        assert "P0301" in context
        assert "1. henger egeskimaradas" in context

    @pytest.mark.asyncio
    async def test_symptom_leg_uses_complaint_type_not_nonexistent_symptom_type(self, rag_service):
        """The unified collection has no ``symptom`` payloads; the similar-case
        leg must ask for ``complaint`` (which exists) instead of silently
        matching nothing.
        """
        from app.services.rag_service import RetrievalSource, VehicleInfo

        rag_service._qdrant = _unified_qdrant_mock(
            {"dtc": UNIFIED_DTC_HITS, "complaint": UNIFIED_COMPLAINT_HITS}
        )

        with (
            patch(
                "app.services.rag_service.embed_text_async", new=AsyncMock(return_value=[0.1] * 768)
            ),
            patch("app.services.rag_service.preprocess_hungarian", side_effect=lambda text: text),
            patch("app.services.rag_service.get_diagnostic_path", new=AsyncMock(return_value={})),
        ):
            context = await rag_service.assemble_context(
                vehicle_info=VehicleInfo(make="Volkswagen", model="Golf", year=2018),
                dtc_codes=["P0301"],
                symptoms="egyenetlen jaratas",
            )

        all_filters = [
            call.kwargs.get("filter_conditions") or {}
            for call in rag_service._qdrant.search.call_args_list
        ]
        requested_types = {filters.get("type") for filters in all_filters}
        assert requested_types == {"dtc", "complaint"}
        assert "symptom" not in requested_types

        assert len(context.symptom_items) == 1
        assert context.symptom_items[0].source == RetrievalSource.QDRANT_COMPLAINT
        # No make filter: NHTSA stores upper-case makes, so an exact-match
        # filter on the user's spelling would silently return nothing.
        complaint_filters = [f for f in all_filters if f.get("type") == "complaint"]
        assert complaint_filters == [{"type": "complaint"}]

    @pytest.mark.asyncio
    async def test_retrieval_never_filters_by_embedding_model_version(self, rag_service):
        """Unified points carry no ``_embedding_model_version`` payload, so
        passing a model version would filter every hit away.
        """
        rag_service._qdrant = _unified_qdrant_mock({"dtc": UNIFIED_DTC_HITS})

        with patch(
            "app.services.rag_service.embed_text_async",
            new=AsyncMock(return_value=[0.1] * 768),
        ):
            await rag_service.retrieve_from_qdrant(query="P0301", type_="dtc")

        _, kwargs = rag_service._qdrant.search.call_args
        assert kwargs["model_version"] is None

    @pytest.mark.asyncio
    async def test_identical_query_does_not_leak_between_type_legs(self, rag_service):
        """Both legs share one collection; with an empty symptom text they also
        share the query string, so the cache key must include the type.
        """
        rag_service._qdrant = _unified_qdrant_mock(
            {"dtc": UNIFIED_DTC_HITS, "complaint": UNIFIED_COMPLAINT_HITS}
        )

        with patch(
            "app.services.rag_service.embed_text_async",
            new=AsyncMock(return_value=[0.1] * 768),
        ):
            dtc_items = await rag_service.retrieve_from_qdrant(query="P0301", type_="dtc")
            complaint_items = await rag_service.retrieve_from_qdrant(
                query="P0301", type_="complaint"
            )

        assert dtc_items[0].content["code"] == "P0301"
        assert complaint_items[0].content["odi_id"] == "11554321"
        assert rag_service._qdrant.search.await_count == 2

    @pytest.mark.asyncio
    async def test_assemble_context_survives_qdrant_failure(self, rag_service):
        """A Qdrant outage must degrade the context, never raise (no 500 on
        /diagnosis/analyze).
        """
        from app.services.rag_service import RAGContext, VehicleInfo

        rag_service._qdrant = _unified_qdrant_mock(error=Exception("Qdrant unavailable"))

        with (
            patch(
                "app.services.rag_service.embed_text_async", new=AsyncMock(return_value=[0.1] * 768)
            ),
            patch("app.services.rag_service.preprocess_hungarian", side_effect=lambda text: text),
            patch("app.services.rag_service.get_diagnostic_path", new=AsyncMock(return_value={})),
        ):
            context = await rag_service.assemble_context(
                vehicle_info=VehicleInfo(make="Volkswagen", model="Golf", year=2018),
                dtc_codes=["P0301"],
                symptoms="egyenetlen jaratas",
            )

        assert isinstance(context, RAGContext)
        assert context.dtc_items == []
        assert context.symptom_items == []

    @pytest.mark.asyncio
    async def test_verify_cross_db_consistency_reports_unified_collection(self, rag_service):
        """The health check must count the collection that is actually queried;
        reporting the empty legacy one is a misleading "ok".
        """
        from app.core.config import settings
        from app.db.qdrant_client import QdrantService

        mock_qdrant = _unified_qdrant_mock()
        mock_qdrant.get_collection_info = AsyncMock(return_value={"points_count": 54652})
        rag_service._qdrant = mock_qdrant

        with patch("app.db.neo4j_models.is_neo4j_available", new=AsyncMock(return_value=True)):
            report = await rag_service.verify_cross_db_consistency()

        mock_qdrant.get_collection_info.assert_awaited_once_with(settings.QDRANT_UNIFIED_COLLECTION)
        assert report["details"]["qdrant"]["collection"] == settings.QDRANT_UNIFIED_COLLECTION
        assert report["details"]["qdrant"]["collection"] not in _LEGACY_COLLECTION_NAMES
        assert report["details"]["qdrant"]["count"] == 54652

    @pytest.mark.asyncio
    async def test_get_context_helper_uses_unified_collection(self, rag_service):
        """The module-level convenience helper follows the same route."""
        from app.core.config import settings
        from app.services.rag_service import get_context

        rag_service._qdrant = _unified_qdrant_mock({"dtc": UNIFIED_DTC_HITS})

        with patch(
            "app.services.rag_service.embed_text_async",
            new=AsyncMock(return_value=[0.1] * 768),
        ):
            results = await get_context("egyenetlen jaratas", top_k=5)

        _, kwargs = rag_service._qdrant.search.call_args
        assert kwargs["filter_conditions"]["type"] == "dtc"
        assert results[0]["content"]["code"] == "P0301"
        assert results[0]["source"] == "qdrant_dtc"

    @pytest.mark.asyncio
    async def test_the_free_text_collection_route_is_gone(self, rag_service):
        """REVERT-GUARD: the ``collection=`` argument WAS the drift's carrier.

        While it existed, any caller could aim a retrieval leg at an
        all-but-empty collection and receive ``[]`` - a legitimate search
        result - so the failure was invisible. It is now a TypeError at the
        call site instead of a silent empty answer at runtime.
        """
        rag_service._qdrant = _unified_qdrant_mock()

        with pytest.raises(TypeError):
            await rag_service.retrieve_from_qdrant(
                query="teszt", collection="dtc_embeddings_hu", top_k=3
            )
        rag_service._qdrant.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_an_unindexed_payload_type_raises_instead_of_returning_empty(self, rag_service):
        """``symptom`` is the real example: the RAG asked for a type the
        collection has none of, and got a plausible empty result."""
        rag_service._qdrant = _unified_qdrant_mock()

        with pytest.raises(ValueError, match="Unknown retrieval type"):
            await rag_service.retrieve_from_qdrant(query="teszt", type_="symptom")
        rag_service._qdrant.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_retrieve_requires_a_payload_type(self, rag_service):
        """No route selected is a programming error, not a silent no-op."""
        with pytest.raises(TypeError):
            await rag_service.retrieve_from_qdrant(query="teszt")


class TestConsistencyServiceCollectionRouting:
    """The admin cross-DB consistency check must count the vectors that exist."""

    @pytest.mark.asyncio
    async def test_dtc_vector_count_reads_unified_collection_with_type_filter(self):
        """REVERT-GUARD: counting the empty ``dtc_embeddings_hu`` collection
        reported 0 vectors and declared a permanent, bogus inconsistency.
        """
        from app.core.config import settings
        from app.services.consistency_service import ConsistencyService

        client = MagicMock()
        client.count.return_value = MagicMock(count=54652)

        with patch("qdrant_client.QdrantClient", return_value=client):
            total = await ConsistencyService()._get_qdrant_vector_count()

        assert total == 54652
        _, kwargs = client.count.call_args
        assert kwargs["collection_name"] == settings.QDRANT_UNIFIED_COLLECTION
        assert kwargs["collection_name"] != "dtc_embeddings_hu"
        condition = kwargs["count_filter"].must[0]
        assert condition.key == "type"
        assert condition.match.value == "dtc"
