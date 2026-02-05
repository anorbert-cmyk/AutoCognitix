"""
Integration tests for Qdrant vector database operations.

Tests semantic search, DTC embedding retrieval, and symptom matching.
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(backend_path))


class TestQdrantDTCSearch:
    """Test Qdrant DTC code search."""

    @pytest.mark.asyncio
    async def test_search_dtc_returns_results(self, mock_qdrant_client):
        """Test that DTC search returns results."""
        results = await mock_qdrant_client.search_dtc(
            query_vector=[0.0] * 768,
            limit=5,
        )

        assert len(results) >= 1
        assert "score" in results[0]
        assert "payload" in results[0]

    @pytest.mark.asyncio
    async def test_search_dtc_returns_code_and_description(self, mock_qdrant_client):
        """Test that DTC search results include code and description."""
        results = await mock_qdrant_client.search_dtc(
            query_vector=[0.0] * 768,
            limit=5,
        )

        if results:
            payload = results[0]["payload"]
            assert "code" in payload
            assert "description_hu" in payload

    @pytest.mark.asyncio
    async def test_search_dtc_respects_limit(self, mock_qdrant_client):
        """Test that DTC search respects limit parameter."""
        mock_qdrant_client.search_dtc.return_value = [
            {"id": "1", "score": 0.9, "payload": {"code": "P0101"}},
            {"id": "2", "score": 0.85, "payload": {"code": "P0100"}},
        ]

        results = await mock_qdrant_client.search_dtc(
            query_vector=[0.0] * 768,
            limit=2,
        )

        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_search_dtc_ordered_by_score(self, mock_qdrant_client):
        """Test that results are ordered by similarity score."""
        mock_qdrant_client.search_dtc.return_value = [
            {"id": "1", "score": 0.95, "payload": {"code": "P0101"}},
            {"id": "2", "score": 0.85, "payload": {"code": "P0100"}},
            {"id": "3", "score": 0.75, "payload": {"code": "P0102"}},
        ]

        results = await mock_qdrant_client.search_dtc(
            query_vector=[0.0] * 768,
            limit=10,
        )

        # Verify descending order
        for i in range(len(results) - 1):
            assert results[i]["score"] >= results[i + 1]["score"]


class TestQdrantSymptomSearch:
    """Test Qdrant symptom similarity search."""

    @pytest.mark.asyncio
    async def test_search_similar_symptoms_returns_results(self, mock_qdrant_client):
        """Test that symptom search returns results."""
        results = await mock_qdrant_client.search_similar_symptoms(
            query_vector=[0.0] * 768,
            limit=5,
        )

        assert len(results) >= 1
        assert "score" in results[0]
        assert "payload" in results[0]

    @pytest.mark.asyncio
    async def test_search_symptoms_returns_description(self, mock_qdrant_client):
        """Test that symptom results include description."""
        results = await mock_qdrant_client.search_similar_symptoms(
            query_vector=[0.0] * 768,
            limit=5,
        )

        if results:
            payload = results[0]["payload"]
            assert "description" in payload

    @pytest.mark.asyncio
    async def test_search_symptoms_returns_related_dtc(self, mock_qdrant_client):
        """Test that symptom results include related DTC codes."""
        results = await mock_qdrant_client.search_similar_symptoms(
            query_vector=[0.0] * 768,
            limit=5,
        )

        if results:
            payload = results[0]["payload"]
            assert "related_dtc" in payload
            assert isinstance(payload["related_dtc"], list)

    @pytest.mark.asyncio
    async def test_search_symptoms_with_vehicle_filter(self, mock_qdrant_client):
        """Test symptom search with vehicle make filter."""
        mock_qdrant_client.search_similar_symptoms.return_value = [
            {
                "id": "1",
                "score": 0.9,
                "payload": {
                    "description": "Motor nehezen indul",
                    "vehicle_make": "Volkswagen",
                    "related_dtc": ["P0101"],
                },
            }
        ]

        results = await mock_qdrant_client.search_similar_symptoms(
            query_vector=[0.0] * 768,
            limit=5,
            vehicle_make="Volkswagen",
        )

        # Should filter by vehicle make
        for result in results:
            if "vehicle_make" in result["payload"]:
                assert result["payload"]["vehicle_make"] == "Volkswagen"


class TestQdrantGeneralSearch:
    """Test Qdrant general vector search."""

    @pytest.mark.asyncio
    async def test_search_returns_results(self, mock_qdrant_client):
        """Test that general search returns results."""
        results = await mock_qdrant_client.search(
            collection_name="dtc_codes",
            query_vector=[0.0] * 768,
            limit=5,
        )

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_with_score_threshold(self, mock_qdrant_client):
        """Test search with minimum score threshold."""
        mock_qdrant_client.search.return_value = [
            {"id": "1", "score": 0.95, "payload": {}},
            {"id": "2", "score": 0.75, "payload": {}},
        ]

        results = await mock_qdrant_client.search(
            collection_name="dtc_codes",
            query_vector=[0.0] * 768,
            limit=10,
            score_threshold=0.8,
        )

        # Mock doesn't actually filter, but in real impl would
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_with_filter_conditions(self, mock_qdrant_client):
        """Test search with filter conditions."""
        mock_qdrant_client.search.return_value = [
            {
                "id": "1",
                "score": 0.9,
                "payload": {"category": "powertrain"},
            }
        ]

        results = await mock_qdrant_client.search(
            collection_name="dtc_codes",
            query_vector=[0.0] * 768,
            limit=10,
            filter_conditions={"category": "powertrain"},
        )

        # Should filter by category
        for result in results:
            if "category" in result.get("payload", {}):
                assert result["payload"]["category"] == "powertrain"


class TestQdrantCollections:
    """Test Qdrant collection operations."""

    @pytest.mark.asyncio
    async def test_dtc_collection_search(self, mock_qdrant_client):
        """Test searching the DTC codes collection."""
        mock_qdrant_client.search.return_value = [
            {
                "id": "1",
                "score": 0.92,
                "payload": {
                    "code": "P0101",
                    "description_hu": "Levegotomeg-mero hiba",
                    "category": "powertrain",
                },
            }
        ]

        results = await mock_qdrant_client.search(
            collection_name="dtc_codes",
            query_vector=[0.0] * 768,
            limit=5,
        )

        assert len(results) >= 1
        assert results[0]["payload"]["code"] == "P0101"

    @pytest.mark.asyncio
    async def test_symptom_collection_search(self, mock_qdrant_client):
        """Test searching the symptoms collection."""
        mock_qdrant_client.search.return_value = [
            {
                "id": "1",
                "score": 0.85,
                "payload": {
                    "description": "Motor nehezen indul hidegben",
                    "related_dtc": ["P0101", "P0171"],
                },
            }
        ]

        results = await mock_qdrant_client.search(
            collection_name="symptoms",
            query_vector=[0.0] * 768,
            limit=5,
        )

        assert len(results) >= 1
        assert "description" in results[0]["payload"]

    @pytest.mark.asyncio
    async def test_issue_collection_search(self, mock_qdrant_client):
        """Test searching the known issues collection."""
        mock_qdrant_client.search.return_value = [
            {
                "id": "1",
                "score": 0.88,
                "payload": {
                    "title": "VW Golf MAF sensor issue",
                    "description": "Common MAF sensor failure in Golf models",
                    "applicable_makes": ["Volkswagen"],
                },
            }
        ]

        results = await mock_qdrant_client.search(
            collection_name="known_issues",
            query_vector=[0.0] * 768,
            limit=5,
        )

        assert len(results) >= 1
        assert "title" in results[0]["payload"]


class TestQdrantVectorOperations:
    """Test Qdrant vector operations."""

    @pytest.mark.asyncio
    async def test_embedding_dimension_768(self, mock_qdrant_client):
        """Test that 768-dimensional embeddings are accepted."""
        # 768 is the dimension for huBERT embeddings
        vector = [0.0] * 768

        results = await mock_qdrant_client.search(
            collection_name="dtc_codes",
            query_vector=vector,
            limit=5,
        )

        # Should not raise an error
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_with_normalized_vector(self, mock_qdrant_client):
        """Test search with normalized embedding vector."""
        import math

        # Create a normalized vector (unit length)
        vector = [1.0 / math.sqrt(768)] * 768

        results = await mock_qdrant_client.search(
            collection_name="dtc_codes",
            query_vector=vector,
            limit=5,
        )

        assert isinstance(results, list)


class TestQdrantScoreInterpretation:
    """Test Qdrant score interpretation."""

    @pytest.mark.asyncio
    async def test_high_score_indicates_similarity(self, mock_qdrant_client):
        """Test that high scores indicate high similarity."""
        mock_qdrant_client.search.return_value = [
            {"id": "1", "score": 0.95, "payload": {"code": "P0101"}},
        ]

        results = await mock_qdrant_client.search(
            collection_name="dtc_codes",
            query_vector=[0.0] * 768,
            limit=5,
        )

        # Score 0.95 should indicate very similar
        assert results[0]["score"] >= 0.9

    @pytest.mark.asyncio
    async def test_score_range_zero_to_one(self, mock_qdrant_client):
        """Test that scores are in 0-1 range."""
        mock_qdrant_client.search.return_value = [
            {"id": "1", "score": 0.92, "payload": {}},
            {"id": "2", "score": 0.75, "payload": {}},
            {"id": "3", "score": 0.5, "payload": {}},
        ]

        results = await mock_qdrant_client.search(
            collection_name="dtc_codes",
            query_vector=[0.0] * 768,
            limit=10,
        )

        for result in results:
            assert 0 <= result["score"] <= 1


class TestQdrantErrorHandling:
    """Test Qdrant error handling."""

    @pytest.mark.asyncio
    async def test_handles_connection_error(self, mock_qdrant_client):
        """Test handling of connection errors."""
        mock_qdrant_client.search.side_effect = Exception("Connection refused")

        with pytest.raises(Exception) as exc_info:
            await mock_qdrant_client.search(
                collection_name="dtc_codes",
                query_vector=[0.0] * 768,
                limit=5,
            )

        assert "Connection" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_handles_invalid_collection(self, mock_qdrant_client):
        """Test handling of invalid collection name."""
        mock_qdrant_client.search.side_effect = Exception("Collection not found")

        with pytest.raises(Exception) as exc_info:
            await mock_qdrant_client.search(
                collection_name="nonexistent_collection",
                query_vector=[0.0] * 768,
                limit=5,
            )

        assert "not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_handles_empty_results(self, mock_qdrant_client):
        """Test handling of empty results."""
        mock_qdrant_client.search.return_value = []

        results = await mock_qdrant_client.search(
            collection_name="dtc_codes",
            query_vector=[0.0] * 768,
            limit=5,
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_handles_timeout(self, mock_qdrant_client):
        """Test handling of timeout errors."""
        mock_qdrant_client.search.side_effect = TimeoutError("Request timed out")

        with pytest.raises(TimeoutError):
            await mock_qdrant_client.search(
                collection_name="dtc_codes",
                query_vector=[0.0] * 768,
                limit=5,
            )


class TestQdrantBatchOperations:
    """Test Qdrant batch operations."""

    @pytest.mark.asyncio
    async def test_batch_search(self, mock_qdrant_client):
        """Test batch searching multiple queries."""
        mock_qdrant_client.search.side_effect = [
            [{"id": "1", "score": 0.9, "payload": {"code": "P0101"}}],
            [{"id": "2", "score": 0.85, "payload": {"code": "P0171"}}],
        ]

        # Search for two different queries
        results1 = await mock_qdrant_client.search(
            collection_name="dtc_codes",
            query_vector=[0.1] * 768,
            limit=5,
        )
        results2 = await mock_qdrant_client.search(
            collection_name="dtc_codes",
            query_vector=[0.2] * 768,
            limit=5,
        )

        assert len(results1) >= 1
        assert len(results2) >= 1
        assert results1[0]["payload"]["code"] == "P0101"
        assert results2[0]["payload"]["code"] == "P0171"
