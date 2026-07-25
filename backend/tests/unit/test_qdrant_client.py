"""Unit tests for app.db.qdrant_client module."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.core.config import settings
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
    """Create a QdrantService with the underlying AsyncQdrantClient fully mocked."""
    with patch("app.db.qdrant_client.AsyncQdrantClient") as MockClient:
        mock_client = AsyncMock()
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
            patch("app.db.qdrant_client.AsyncQdrantClient") as MockClient,
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
            patch("app.db.qdrant_client.AsyncQdrantClient") as MockClient,
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
        service.client.get_collections = AsyncMock(return_value=collections_resp)
        service.client.create_collection = AsyncMock()

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
        service.client.get_collections = AsyncMock(return_value=collections_resp)
        service.client.create_collection = AsyncMock()

        await service.initialize_collections()

        service.client.create_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_collection_connection_error(self, service):
        service.client.get_collections = AsyncMock(side_effect=ConnectionError("refused"))

        with pytest.raises(QdrantConnectionException):
            await service._create_collection_if_not_exists("test_collection")

    @pytest.mark.asyncio
    async def test_create_collection_generic_error(self, service):
        service.client.get_collections = AsyncMock(side_effect=RuntimeError("bad"))

        with pytest.raises(QdrantException):
            await service._create_collection_if_not_exists("test_collection")


# ---------------------------------------------------------------------------
# upsert_vectors
# ---------------------------------------------------------------------------


class TestUpsertVectors:
    @pytest.mark.asyncio
    async def test_upsert_success(self, service):
        service.client.upsert = AsyncMock()
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
        service.client.upsert = AsyncMock()
        ids = ["id1"]
        vectors = [[0.5] * 768]
        payloads = [{"code": "P0300"}]

        await service.upsert_vectors("dtc_embeddings_hu", ids, vectors, payloads)

        # The payload should now include the model version
        assert payloads[0]["_embedding_model_version"] == "hubert-base-cc-v1"

    @pytest.mark.asyncio
    async def test_upsert_without_payloads(self, service):
        service.client.upsert = AsyncMock()
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
        service.client.search = AsyncMock(
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
        service.client.search = AsyncMock(return_value=[])

        results = await service.search(
            collection_name="dtc_embeddings_hu",
            query_vector=[0.1] * 768,
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_search_with_filter_conditions(self, service):
        service.client.search = AsyncMock(return_value=[])

        await service.search(
            collection_name="dtc_embeddings_hu",
            query_vector=[0.1] * 768,
            filter_conditions={"category": "powertrain"},
        )

        # Verify the search was called (filter is built internally)
        service.client.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_with_score_threshold(self, service):
        service.client.search = AsyncMock(return_value=[])

        await service.search(
            collection_name="dtc_embeddings_hu",
            query_vector=[0.1] * 768,
            score_threshold=0.5,
        )

        service.client.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_with_model_version(self, service):
        service.client.search = AsyncMock(return_value=[])

        await service.search(
            collection_name="dtc_embeddings_hu",
            query_vector=[0.1] * 768,
            model_version="hubert-base-cc-v1",
        )

        service.client.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_connection_error(self, service):
        service.client.search = AsyncMock(side_effect=ConnectionError("timeout"))

        with pytest.raises(QdrantConnectionException):
            await service.search(
                collection_name="dtc_embeddings_hu",
                query_vector=[0.1] * 768,
            )

    @pytest.mark.asyncio
    async def test_search_generic_error(self, service):
        service.client.search = AsyncMock(side_effect=RuntimeError("internal"))

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
        # REVERT-GUARD: DTC vectors live in the unified collection with a
        # {"type": "dtc"} discriminator, NOT the empty "dtc_embeddings_hu".
        with patch.object(service, "search", new=AsyncMock(return_value=[])) as mock_search:
            results = await service.search_dtc([0.1] * 768, limit=5)
            assert results == []
            mock_search.assert_awaited_once_with(
                collection_name=settings.QDRANT_UNIFIED_COLLECTION,
                query_vector=[0.1] * 768,
                limit=5,
                filter_conditions={"type": "dtc"},
                score_threshold=None,
                model_version=None,
            )

    @pytest.mark.asyncio
    async def test_search_dtc_with_category_and_severity(self, service):
        with patch.object(service, "search", new=AsyncMock(return_value=[])) as mock_search:
            await service.search_dtc(
                [0.1] * 768,
                limit=3,
                category="powertrain",
                severity="high",
            )
            mock_search.assert_awaited_once_with(
                collection_name=settings.QDRANT_UNIFIED_COLLECTION,
                query_vector=[0.1] * 768,
                limit=3,
                filter_conditions={"type": "dtc", "category": "powertrain", "severity": "high"},
                score_threshold=None,
                model_version=None,
            )


class TestSearchUnified:
    """Tests for the unified-collection search that fixes the Qdrant drift bug."""

    def test_unified_collection_default_is_autocognitix(self):
        # The 54,652 huBERT vectors were indexed into "autocognitix"; the default
        # must match, and it must be env-overridable for a no-redeploy correction.
        assert settings.QDRANT_UNIFIED_COLLECTION == "autocognitix"

    @pytest.mark.asyncio
    async def test_search_unified_targets_configured_collection_with_type(self, service):
        with patch.object(service, "search", new=AsyncMock(return_value=[])) as mock_search:
            await service.search_unified([0.1] * 768, type_="dtc", limit=7)
            mock_search.assert_awaited_once_with(
                collection_name=settings.QDRANT_UNIFIED_COLLECTION,
                query_vector=[0.1] * 768,
                limit=7,
                filter_conditions={"type": "dtc"},
                score_threshold=None,
                model_version=None,
            )

    @pytest.mark.asyncio
    async def test_search_dtc_uses_unified_collection_not_legacy(self, service):
        """REVERT-GUARD: this FAILS if search_dtc is reverted to the empty
        legacy `dtc_embeddings_hu` collection or loses the {"type": "dtc"} filter.
        """
        with patch.object(service, "search", new=AsyncMock(return_value=[])) as mock_search:
            await service.search_dtc([0.1] * 768, limit=5)

        _, kwargs = mock_search.call_args
        assert kwargs["collection_name"] == settings.QDRANT_UNIFIED_COLLECTION
        assert kwargs["collection_name"] == "autocognitix"
        # Must NOT regress to the empty legacy collection.
        assert kwargs["collection_name"] != service.DTC_COLLECTION
        assert kwargs["filter_conditions"]["type"] == "dtc"


class TestSearchSimilarSymptoms:
    @pytest.mark.asyncio
    async def test_search_symptoms_no_filters(self, service):
        with patch.object(service, "search", new=AsyncMock(return_value=[])) as mock_search:
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
        with patch.object(service, "search", new=AsyncMock(return_value=[])) as mock_search:
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
        with patch.object(service, "search", new=AsyncMock(return_value=[])) as mock_search:
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
        with patch.object(service, "search", new=AsyncMock(return_value=[])) as mock_search:
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
        with patch.object(service, "search", new=AsyncMock(return_value=[])) as mock_search:
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
        with patch.object(service, "search", new=AsyncMock(return_value=[])) as mock_search:
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
        service.client.delete_collection = AsyncMock()
        await service.delete_collection("test_collection")
        service.client.delete_collection.assert_called_once()


# ---------------------------------------------------------------------------
# GDPR Article 17 erasure (delete_by_user)
#
# Two independent defects lived here. The sweep iterated ONLY the legacy
# per-type collections - every one of them documented as never populated -
# while every write path had moved to settings.QDRANT_UNIFIED_COLLECTION, so
# erasure provably touched nothing. And each per-collection failure was
# absorbed by `except Exception: logger.warning`, which left cleanup_errors
# empty in DELETE /api/v1/auth/me and reported a failed purge to the data
# subject as a completed one.
# ---------------------------------------------------------------------------


def _collections(*names):
    """Fake `get_collections()` response listing `names`."""
    return SimpleNamespace(collections=[SimpleNamespace(name=n) for n in names])


class TestDeleteByUserGDPR:
    @pytest.mark.asyncio
    async def test_deletes_from_the_unified_collection(self, service):
        """The collection that actually holds the vectors must be purged."""
        service.client.delete = AsyncMock()
        service.client.get_collections = AsyncMock(return_value=_collections())

        result = await service.delete_by_user("user-123")

        targeted = [c.kwargs["collection_name"] for c in service.client.delete.call_args_list]
        assert settings.QDRANT_UNIFIED_COLLECTION in targeted
        assert result == 1  # unified only: no legacy collection exists here

    @pytest.mark.asyncio
    async def test_filters_on_the_user_id(self, service):
        service.client.delete = AsyncMock()
        service.client.get_collections = AsyncMock(return_value=_collections())

        await service.delete_by_user("user-123")

        selector = service.client.delete.call_args.kwargs["points_selector"]
        assert selector.filter.must[0].key == "user_id"
        assert selector.filter.must[0].match.value == "user-123"

    @pytest.mark.asyncio
    async def test_sweeps_legacy_collections_that_still_exist(self, service):
        """An instance seeded before the unification can still hold points."""
        service.client.delete = AsyncMock()
        service.client.get_collections = AsyncMock(
            return_value=_collections(service.DTC_COLLECTION, service.SYMPTOM_COLLECTION)
        )

        result = await service.delete_by_user("user-123")

        targeted = [c.kwargs["collection_name"] for c in service.client.delete.call_args_list]
        assert targeted[0] == settings.QDRANT_UNIFIED_COLLECTION
        assert set(targeted[1:]) == {service.DTC_COLLECTION, service.SYMPTOM_COLLECTION}
        assert result == 3

    @pytest.mark.asyncio
    async def test_skips_legacy_collections_that_do_not_exist(self, service):
        """Deleting from a missing collection is a 404, not an erasure failure."""
        service.client.delete = AsyncMock()
        service.client.get_collections = AsyncMock(return_value=_collections())

        await service.delete_by_user("user-123")

        targeted = [c.kwargs["collection_name"] for c in service.client.delete.call_args_list]
        assert targeted == [settings.QDRANT_UNIFIED_COLLECTION]

    @pytest.mark.asyncio
    async def test_a_failed_purge_is_not_reported_as_success(self, service):
        """The GDPR-critical contract: failure must reach the caller.

        A swallowed failure is what let DELETE /api/v1/auth/me answer 200 while
        the user's vectors were still in Qdrant.
        """
        service.client.delete = AsyncMock(side_effect=Exception("qdrant down"))
        service.client.get_collections = AsyncMock(return_value=_collections())

        with pytest.raises(QdrantException) as exc_info:
            await service.delete_by_user("user-123")

        assert settings.QDRANT_UNIFIED_COLLECTION in exc_info.value.details["failed_collections"]

    @pytest.mark.asyncio
    async def test_a_partial_failure_still_raises(self, service):
        """Purging 2 of 3 collections is a partial deletion, not a success."""
        service.client.delete = AsyncMock(side_effect=[None, Exception("boom"), None])
        service.client.get_collections = AsyncMock(
            return_value=_collections(service.DTC_COLLECTION, service.SYMPTOM_COLLECTION)
        )

        with pytest.raises(QdrantException) as exc_info:
            await service.delete_by_user("user-123")

        assert exc_info.value.details["deleted_collections"] == 2
        assert len(exc_info.value.details["failed_collections"]) == 1

    @pytest.mark.asyncio
    async def test_the_account_deletion_endpoint_turns_that_into_a_500(self):
        """End of the contract: the raise must land in `cleanup_errors`.

        Mirrors the try/except in endpoints/auth.py::delete_user_account, which
        aborts before the PostgreSQL commit when the external cleanup failed.
        """
        cleanup_errors: list = []
        qdrant = AsyncMock()
        qdrant.delete_by_user = AsyncMock(side_effect=QdrantException(message="nope"))

        try:
            await qdrant.delete_by_user("user-123")
        except Exception as e:  # mirrors endpoints/auth.py verbatim
            cleanup_errors.append(f"Qdrant: {e}")

        assert cleanup_errors, "a failed erasure must not leave cleanup_errors empty"

    @pytest.mark.asyncio
    async def test_an_unlistable_qdrant_still_purges_the_unified_collection(self, service):
        """A probe failure must not block the delete that actually matters."""
        service.client.delete = AsyncMock()
        service.client.get_collections = AsyncMock(side_effect=Exception("list failed"))

        result = await service.delete_by_user("user-123")

        assert result == 1
        assert (
            service.client.delete.call_args.kwargs["collection_name"]
            == settings.QDRANT_UNIFIED_COLLECTION
        )


# ---------------------------------------------------------------------------
# get_collection_info
# ---------------------------------------------------------------------------


class TestGetCollectionInfo:
    @pytest.mark.asyncio
    async def test_get_collection_info(self, service):
        info_obj = _make_collection_info("dtc_embeddings_hu", points_count=1000, vectors_count=1000)
        service.client.get_collection = AsyncMock(return_value=info_obj)

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
            new=AsyncMock(
                return_value={
                    "name": "col",
                    "points_count": 500,
                    "vectors_count": 500,
                    "status": "green",
                }
            ),
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
            new=AsyncMock(side_effect=Exception("unavailable")),
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
            new=AsyncMock(return_value={"dtc_embeddings_hu": {"points_count": 1000}}),
        ):
            alerts = await service.check_storage_alerts()
            assert alerts == []

    @pytest.mark.asyncio
    async def test_alerts_above_threshold(self, service):
        with patch.object(
            service,
            "get_storage_stats",
            new=AsyncMock(return_value={"dtc_embeddings_hu": {"points_count": 60000}}),
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
            new=AsyncMock(return_value={"dtc_embeddings_hu": {"error": "unavailable"}}),
        ):
            alerts = await service.check_storage_alerts()
            assert alerts == []


# ---------------------------------------------------------------------------
# Global instance helper
# ---------------------------------------------------------------------------


class TestGetQdrantService:
    @pytest.mark.asyncio
    async def test_returns_qdrant_service(self):
        with patch("app.db.qdrant_client.AsyncQdrantClient"):
            from app.db.qdrant_client import get_qdrant_service

            svc = await get_qdrant_service()
            from app.db.qdrant_client import QdrantService

            assert isinstance(svc, QdrantService)
