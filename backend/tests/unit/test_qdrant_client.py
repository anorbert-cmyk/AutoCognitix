"""Unit tests for app.db.qdrant_client module."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.core.exceptions import QdrantConnectionException, QdrantException


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_search_result(id_, score, payload):
    """Create a fake Qdrant ScoredPoint."""
    r = SimpleNamespace()
    r.id = id_
    r.score = score
    r.payload = payload
    return r


def _make_collection_info(name, points_count, status="green", vectors_count=0):
    obj = SimpleNamespace()
    obj.name = name
    obj.points_count = points_count
    obj.status = status
    obj.indexed_vectors_count = vectors_count
    return obj


# ---------------------------------------------------------------------------
# Fixture: QdrantService with a mocked client
# ---------------------------------------------------------------------------


@pytest.fixture
def service():
    """Create a QdrantService with the underlying QdrantClient fully mocked."""
    with patch("app.db.qdrant_client.QdrantClient") as MockClient:
        mock_client = MagicMock()
        MockClient.return_value = mock_client

        from app.db.qdrant_client import QdrantService

        svc = QdrantService()
        # Ensure mock_client is what we injected
        assert svc.client is mock_client
        yield svc


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInit:
    def test_init_with_cloud_url(self):
        with (
            patch("app.db.qdrant_client.settings") as mock_settings,
            patch("app.db.qdrant_client.QdrantClient") as MockClient,
        ):
            mock_settings.QDRANT_URL = "https://cloud.qdrant.io:6333"
            mock_settings.QDRANT_API_KEY = "test-key"
            mock_settings.EMBEDDING_DIMENSION = 768

            from app.db.qdrant_client import QdrantService

            svc = QdrantService()

            MockClient.assert_called_once_with(
                url="https://cloud.qdrant.io:6333",
                api_key="test-key",
            )
            assert svc.vector_size == 768

    def test_init_with_local(self):
        with (
            patch("app.db.qdrant_client.settings") as mock_settings,
            patch("app.db.qdrant_client.QdrantClient") as MockClient,
        ):
            mock_settings.QDRANT_URL = ""  # falsy → local
            mock_settings.QDRANT_API_KEY = None
            mock_settings.QDRANT_HOST = "localhost"
            mock_settings.QDRANT_PORT = 6333
            mock_settings.EMBEDDING_DIMENSION = 768

            from app.db.qdrant_client import QdrantService

            svc = QdrantService()

            MockClient.assert_called_once_with(
                host="localhost",
                port=6333,
                prefer_grpc=True,
            )
            assert svc.vector_size == 768

    def test_collection_constants(self, service):
        assert service.DTC_COLLECTION == "dtc_embeddings_hu"
        assert service.SYMPTOM_COLLECTION == "symptom_embeddings_hu"
        assert service.COMPONENT_COLLECTION == "component_embeddings_hu"
        assert service.REPAIR_COLLECTION == "repair_embeddings_hu"
        assert service.ISSUE_COLLECTION == "known_issue_embeddings_hu"

    def test_expected_dimension(self, service):
        assert service.EXPECTED_DIMENSION == 768


# ---------------------------------------------------------------------------
# initialize_collections
# ---------------------------------------------------------------------------


class TestInitializeCollections:
    @pytest.mark.asyncio
    async def test_creates_missing_collections(self, service):
        # No collections exist yet
        collections_resp = SimpleNamespace(collections=[])
        service.client.get_collections = MagicMock(return_value=collections_resp)
        service.client.create_collection = MagicMock()

        await service.initialize_collections()

        assert service.client.create_collection.call_count == 5

    @pytest.mark.asyncio
    async def test_skips_existing_collections(self, service):
        existing = [
            SimpleNamespace(name="dtc_embeddings_hu"),
            SimpleNamespace(name="symptom_embeddings_hu"),
            SimpleNamespace(name="component_embeddings_hu"),
            SimpleNamespace(name="repair_embeddings_hu"),
            SimpleNamespace(name="known_issue_embeddings_hu"),
        ]
        collections_resp = SimpleNamespace(collections=existing)
        service.client.get_collections = MagicMock(return_value=collections_resp)
        service.client.create_collection = MagicMock()

        await service.initialize_collections()

        service.client.create_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_collection_connection_error(self, service):
        service.client.get_collections = MagicMock(side_effect=ConnectionError("refused"))

        with pytest.raises(QdrantConnectionException):
            await service._create_collection_if_not_exists("test_collection")

    @pytest.mark.asyncio
    async def test_create_collection_generic_error(self, service):
        service.client.get_collections = MagicMock(side_effect=RuntimeError("bad"))

        with pytest.raises(QdrantException):
            await service._create_collection_if_not_exists("test_collection")


# ---------------------------------------------------------------------------
# upsert_vectors
# ---------------------------------------------------------------------------


class TestUpsertVectors:
    @pytest.mark.asyncio
    async def test_upsert_success(self, service):
        service.client.upsert = MagicMock()
        ids = ["id1", "id2"]
        vectors = [[0.1] * 768, [0.2] * 768]
        payloads = [{"code": "P0300"}, {"code": "P0301"}]

        await service.upsert_vectors("dtc_embeddings_hu", ids, vectors, payloads)

        service.client.upsert.assert_called_once()
        call_kwargs = service.client.upsert.call_args
        # Points should have model version injected
        points = call_kwargs.kwargs.get("points") or call_kwargs[1].get("points")
        if points is None:
            # positional or via to_thread wrapper
            pass

    @pytest.mark.asyncio
    async def test_upsert_injects_model_version(self, service):
        service.client.upsert = MagicMock()
        ids = ["id1"]
        vectors = [[0.5] * 768]
        payloads = [{"code": "P0300"}]

        await service.upsert_vectors("dtc_embeddings_hu", ids, vectors, payloads)

        # The payload should now include the model version
        assert payloads[0]["_embedding_model_version"] == "hubert-base-cc-v1"

    @pytest.mark.asyncio
    async def test_upsert_without_payloads(self, service):
        service.client.upsert = MagicMock()
        ids = ["id1"]
        vectors = [[0.1] * 768]

        await service.upsert_vectors("dtc_embeddings_hu", ids, vectors, payloads=None)

        service.client.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_upsert_dimension_mismatch_raises(self, service):
        ids = ["id1"]
        vectors = [[0.1] * 100]  # wrong dimension

        with pytest.raises(ValueError, match="Vector dimension mismatch"):
            await service.upsert_vectors("dtc_embeddings_hu", ids, vectors)


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


class TestSearch:
    @pytest.mark.asyncio
    async def test_search_returns_results(self, service):
        service.client.search = MagicMock(
            return_value=[
                _make_search_result("id1", 0.95, {"code": "P0300"}),
                _make_search_result("id2", 0.88, {"code": "P0301"}),
            ]
        )

        results = await service.search(
            collection_name="dtc_embeddings_hu",
            query_vector=[0.1] * 768,
            limit=5,
        )

        assert len(results) == 2
        assert results[0]["id"] == "id1"
        assert results[0]["score"] == 0.95
        assert results[0]["payload"] == {"code": "P0300"}
        assert results[1]["id"] == "id2"

    @pytest.mark.asyncio
    async def test_search_empty_results(self, service):
        service.client.search = MagicMock(return_value=[])

        results = await service.search(
            collection_name="dtc_embeddings_hu",
            query_vector=[0.1] * 768,
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_search_with_filter_conditions(self, service):
        service.client.search = MagicMock(return_value=[])

        await service.search(
            collection_name="dtc_embeddings_hu",
            query_vector=[0.1] * 768,
            filter_conditions={"category": "powertrain"},
        )

        # Verify the search was called (filter is built internally)
        service.client.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_with_score_threshold(self, service):
        service.client.search = MagicMock(return_value=[])

        await service.search(
            collection_name="dtc_embeddings_hu",
            query_vector=[0.1] * 768,
            score_threshold=0.5,
        )

        service.client.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_with_model_version(self, service):
        service.client.search = MagicMock(return_value=[])

        await service.search(
            collection_name="dtc_embeddings_hu",
            query_vector=[0.1] * 768,
            model_version="hubert-base-cc-v1",
        )

        service.client.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_connection_error(self, service):
        service.client.search = MagicMock(side_effect=ConnectionError("timeout"))

        with pytest.raises(QdrantConnectionException):
            await service.search(
                collection_name="dtc_embeddings_hu",
                query_vector=[0.1] * 768,
            )

    @pytest.mark.asyncio
    async def test_search_generic_error(self, service):
        service.client.search = MagicMock(side_effect=RuntimeError("internal"))

        with pytest.raises(QdrantException):
            await service.search(
                collection_name="dtc_embeddings_hu",
                query_vector=[0.1] * 768,
            )


# ---------------------------------------------------------------------------
# Specialised search methods
# ---------------------------------------------------------------------------


class TestSearchDTC:
    @pytest.mark.asyncio
    async def test_search_dtc_no_filters(self, service):
        with patch.object(service, "search", return_value=[]) as mock_search:
            results = await service.search_dtc([0.1] * 768, limit=5)
            assert results == []
            mock_search.assert_awaited_once_with(
                collection_name="dtc_embeddings_hu",
                query_vector=[0.1] * 768,
                limit=5,
                filter_conditions=None,
                model_version=None,
            )

    @pytest.mark.asyncio
    async def test_search_dtc_with_category_and_severity(self, service):
        with patch.object(service, "search", return_value=[]) as mock_search:
            await service.search_dtc(
                [0.1] * 768,
                limit=3,
                category="powertrain",
                severity="high",
            )
            mock_search.assert_awaited_once_with(
                collection_name="dtc_embeddings_hu",
                query_vector=[0.1] * 768,
                limit=3,
                filter_conditions={"category": "powertrain", "severity": "high"},
                model_version=None,
            )


class TestSearchSimilarSymptoms:
    @pytest.mark.asyncio
    async def test_search_symptoms_no_filters(self, service):
        with patch.object(service, "search", return_value=[]) as mock_search:
            await service.search_similar_symptoms([0.2] * 768, limit=10)
            mock_search.assert_awaited_once_with(
                collection_name="symptom_embeddings_hu",
                query_vector=[0.2] * 768,
                limit=10,
                filter_conditions=None,
                model_version=None,
            )

    @pytest.mark.asyncio
    async def test_search_symptoms_with_make(self, service):
        with patch.object(service, "search", return_value=[]) as mock_search:
            await service.search_similar_symptoms([0.2] * 768, limit=5, vehicle_make="VW")
            mock_search.assert_awaited_once_with(
                collection_name="symptom_embeddings_hu",
                query_vector=[0.2] * 768,
                limit=5,
                filter_conditions={"vehicle_make": "VW"},
                model_version=None,
            )


class TestSearchComponents:
    @pytest.mark.asyncio
    async def test_search_components_no_filters(self, service):
        with patch.object(service, "search", return_value=[]) as mock_search:
            await service.search_components([0.3] * 768, limit=5)
            mock_search.assert_awaited_once_with(
                collection_name="component_embeddings_hu",
                query_vector=[0.3] * 768,
                limit=5,
                filter_conditions=None,
                model_version=None,
            )

    @pytest.mark.asyncio
    async def test_search_components_with_system(self, service):
        with patch.object(service, "search", return_value=[]) as mock_search:
            await service.search_components([0.3] * 768, system="engine")
            mock_search.assert_awaited_once_with(
                collection_name="component_embeddings_hu",
                query_vector=[0.3] * 768,
                limit=10,
                filter_conditions={"system": "engine"},
                model_version=None,
            )


class TestSearchRepairs:
    @pytest.mark.asyncio
    async def test_search_repairs_no_filters(self, service):
        with patch.object(service, "search", return_value=[]) as mock_search:
            await service.search_repairs([0.4] * 768, limit=3)
            mock_search.assert_awaited_once_with(
                collection_name="repair_embeddings_hu",
                query_vector=[0.4] * 768,
                limit=3,
                filter_conditions=None,
                model_version=None,
            )

    @pytest.mark.asyncio
    async def test_search_repairs_with_difficulty(self, service):
        with patch.object(service, "search", return_value=[]) as mock_search:
            await service.search_repairs([0.4] * 768, difficulty="professional")
            mock_search.assert_awaited_once_with(
                collection_name="repair_embeddings_hu",
                query_vector=[0.4] * 768,
                limit=10,
                filter_conditions={"difficulty": "professional"},
                model_version=None,
            )


# ---------------------------------------------------------------------------
# delete operations
# ---------------------------------------------------------------------------


class TestDeleteOperations:
    @pytest.mark.asyncio
    async def test_delete_collection(self, service):
        service.client.delete_collection = MagicMock()
        await service.delete_collection("test_collection")
        service.client.delete_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_by_user_processes_all_collections(self, service):
        service.client.delete = MagicMock()
        result = await service.delete_by_user("user-123")
        assert result == 5  # 5 collections
        assert service.client.delete.call_count == 5

    @pytest.mark.asyncio
    async def test_delete_by_user_continues_on_error(self, service):
        # First call fails, rest succeed
        service.client.delete = MagicMock(side_effect=[Exception("fail"), None, None, None, None])
        result = await service.delete_by_user("user-123")
        assert result == 4  # 4 successful out of 5


# ---------------------------------------------------------------------------
# get_collection_info
# ---------------------------------------------------------------------------


class TestGetCollectionInfo:
    @pytest.mark.asyncio
    async def test_get_collection_info(self, service):
        info_obj = _make_collection_info("dtc_embeddings_hu", points_count=1000, vectors_count=1000)
        service.client.get_collection = MagicMock(return_value=info_obj)

        info = await service.get_collection_info("dtc_embeddings_hu")
        assert info["name"] == "dtc_embeddings_hu"
        assert info["points_count"] == 1000
        assert info["vectors_count"] == 1000
        assert info["status"] == "green"


# ---------------------------------------------------------------------------
# get_storage_stats
# ---------------------------------------------------------------------------


class TestGetStorageStats:
    @pytest.mark.asyncio
    async def test_get_storage_stats_success(self, service):
        _make_collection_info("col", points_count=500, vectors_count=500)
        with patch.object(
            service,
            "get_collection_info",
            return_value={
                "name": "col",
                "points_count": 500,
                "vectors_count": 500,
                "status": "green",
            },
        ):
            stats = await service.get_storage_stats()
            assert len(stats) == 5
            for coll_stats in stats.values():
                assert coll_stats["points_count"] == 500

    @pytest.mark.asyncio
    async def test_get_storage_stats_handles_errors(self, service):
        with patch.object(
            service,
            "get_collection_info",
            side_effect=Exception("unavailable"),
        ):
            stats = await service.get_storage_stats()
            assert len(stats) == 5
            for coll_stats in stats.values():
                assert coll_stats == {"error": "unavailable"}


# ---------------------------------------------------------------------------
# check_storage_alerts
# ---------------------------------------------------------------------------


class TestCheckStorageAlerts:
    @pytest.mark.asyncio
    async def test_no_alerts_below_threshold(self, service):
        with patch.object(
            service,
            "get_storage_stats",
            return_value={
                "dtc_embeddings_hu": {"points_count": 1000},
            },
        ):
            alerts = await service.check_storage_alerts()
            assert alerts == []

    @pytest.mark.asyncio
    async def test_alerts_above_threshold(self, service):
        with patch.object(
            service,
            "get_storage_stats",
            return_value={
                "dtc_embeddings_hu": {"points_count": 60000},
            },
        ):
            alerts = await service.check_storage_alerts()
            assert len(alerts) == 1
            assert alerts[0]["collection"] == "dtc_embeddings_hu"
            assert alerts[0]["count"] == 60000
            assert alerts[0]["severity"] == "warning"

    @pytest.mark.asyncio
    async def test_alerts_skip_error_entries(self, service):
        with patch.object(
            service,
            "get_storage_stats",
            return_value={
                "dtc_embeddings_hu": {"error": "unavailable"},
            },
        ):
            alerts = await service.check_storage_alerts()
            assert alerts == []


# ---------------------------------------------------------------------------
# Global instance helper
# ---------------------------------------------------------------------------


class TestGetQdrantService:
    @pytest.mark.asyncio
    async def test_returns_qdrant_service(self):
        with patch("app.db.qdrant_client.QdrantClient"):
            from app.db.qdrant_client import get_qdrant_service

            svc = await get_qdrant_service()
            from app.db.qdrant_client import QdrantService

            assert isinstance(svc, QdrantService)
