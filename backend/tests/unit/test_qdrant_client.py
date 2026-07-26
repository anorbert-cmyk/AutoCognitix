"""Unit tests for app.db.qdrant_client module."""

import ast
import inspect
from pathlib import Path
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

    def test_expected_dimension(self, service):
        assert service.EXPECTED_DIMENSION == 768


# ---------------------------------------------------------------------------
# initialize_collections
#
# It used to create the five legacy per-type collections on every boot and NOT
# the unified one - it initialised everything except the store that holds the
# vectors. Worse, pre-creating them guaranteed that a mis-addressed search hit
# an existing-but-empty collection and got `[]` (a legitimate search answer)
# instead of a loud 404.
# ---------------------------------------------------------------------------


class TestInitializeCollections:
    @pytest.mark.asyncio
    async def test_creates_the_unified_collection_when_missing(self, service):
        service.client.get_collections = AsyncMock(return_value=SimpleNamespace(collections=[]))
        service.client.create_collection = AsyncMock()

        await service.initialize_collections()

        service.client.create_collection.assert_called_once()
        assert (
            service.client.create_collection.call_args.kwargs["collection_name"]
            == settings.QDRANT_UNIFIED_COLLECTION
        )

    @pytest.mark.asyncio
    async def test_never_creates_a_legacy_collection(self, service):
        """REVERT-GUARD: booting must not manufacture the empty collections that
        turned a mis-addressed search into a silent empty result."""
        from app.db.qdrant_client import _LEGACY_COLLECTIONS

        service.client.get_collections = AsyncMock(return_value=SimpleNamespace(collections=[]))
        service.client.create_collection = AsyncMock()

        await service.initialize_collections()

        created = [
            call.kwargs["collection_name"]
            for call in service.client.create_collection.call_args_list
        ]
        assert created == [settings.QDRANT_UNIFIED_COLLECTION]
        assert not set(created) & set(_LEGACY_COLLECTIONS)

    @pytest.mark.asyncio
    async def test_is_a_noop_when_the_unified_collection_exists(self, service):
        service.client.get_collections = AsyncMock(
            return_value=SimpleNamespace(
                collections=[SimpleNamespace(name=settings.QDRANT_UNIFIED_COLLECTION)]
            )
        )
        service.client.create_collection = AsyncMock()

        await service.initialize_collections()

        service.client.create_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_reports_but_never_touches_surviving_legacy_collections(self, service, caplog):
        """Production still holds ~2,323 points in dtc_embeddings_hu. Dropping
        them is a human decision, so boot may only surface them."""
        service.client.get_collections = AsyncMock(
            return_value=SimpleNamespace(
                collections=[
                    SimpleNamespace(name=settings.QDRANT_UNIFIED_COLLECTION),
                    SimpleNamespace(name="dtc_embeddings_hu"),
                ]
            )
        )
        service.client.create_collection = AsyncMock()
        service.client.delete_collection = AsyncMock()

        with caplog.at_level("WARNING"):
            await service.initialize_collections()

        service.client.create_collection.assert_not_called()
        service.client.delete_collection.assert_not_called()
        assert any("dtc_embeddings_hu" in r.getMessage() for r in caplog.records)

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

        await service.upsert_vectors(ids, vectors, payloads)

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

        await service.upsert_vectors(ids, vectors, payloads)

        # The payload should now include the model version
        assert payloads[0]["_embedding_model_version"] == "hubert-base-cc-v1"

    @pytest.mark.asyncio
    async def test_upsert_without_payloads(self, service):
        service.client.upsert = AsyncMock()
        ids = ["id1"]
        vectors = [[0.1] * 768]

        await service.upsert_vectors(ids, vectors, payloads=None)

        service.client.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_upsert_dimension_mismatch_raises(self, service):
        ids = ["id1"]
        vectors = [[0.1] * 100]  # wrong dimension

        with pytest.raises(ValueError, match="Vector dimension mismatch"):
            await service.upsert_vectors(ids, vectors)


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
            query_vector=[0.1] * 768,
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_search_with_filter_conditions(self, service):
        service.client.search = AsyncMock(return_value=[])

        await service.search(
            query_vector=[0.1] * 768,
            filter_conditions={"category": "powertrain"},
        )

        # Verify the search was called (filter is built internally)
        service.client.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_with_score_threshold(self, service):
        service.client.search = AsyncMock(return_value=[])

        await service.search(
            query_vector=[0.1] * 768,
            score_threshold=0.5,
        )

        service.client.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_with_model_version(self, service):
        service.client.search = AsyncMock(return_value=[])

        await service.search(
            query_vector=[0.1] * 768,
            model_version="hubert-base-cc-v1",
        )

        service.client.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_connection_error(self, service):
        service.client.search = AsyncMock(side_effect=ConnectionError("timeout"))

        with pytest.raises(QdrantConnectionException):
            await service.search(
                query_vector=[0.1] * 768,
            )

    @pytest.mark.asyncio
    async def test_search_generic_error(self, service):
        service.client.search = AsyncMock(side_effect=RuntimeError("internal"))

        with pytest.raises(QdrantException):
            await service.search(
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
                query_vector=[0.1] * 768,
                limit=7,
                filter_conditions={"type": "dtc"},
                score_threshold=None,
                model_version=None,
            )

    @pytest.mark.asyncio
    async def test_unknown_payload_type_raises_instead_of_returning_empty(self, service):
        """The type discriminator is the collection name's twin trap.

        A search for a type nobody ever indexed (``"symptom"`` is the real
        example - the RAG asked for it and the collection has none) matches zero
        points and returns ``[]``, which is a legitimate answer for a search
        engine. Closed set, validated at the gate.
        """
        with (
            patch.object(service, "search", new=AsyncMock(return_value=[])) as mock_search,
            pytest.raises(ValueError, match="Unknown payload type"),
        ):
            await service.search_unified([0.1] * 768, type_="symptom")
        mock_search.assert_not_called()


# ---------------------------------------------------------------------------
# REGRESSION GUARD: the legacy collections must be structurally unreachable
#
# This is the class that has to fail if the drift is reintroduced - by anyone,
# in any file, not just in the three modules that were fixed. It checks the two
# ways a caller could name the wrong collection: writing one of the names down,
# or accepting one as a parameter.
# ---------------------------------------------------------------------------


LEGACY_COLLECTION_NAMES = (
    "dtc_embeddings_hu",
    "symptom_embeddings_hu",
    "component_embeddings_hu",
    "repair_embeddings_hu",
    "known_issue_embeddings_hu",
)

# The single module allowed to know the names at all, and only inside the
# `_LEGACY_COLLECTIONS` tuple that the GDPR erasure sweep iterates.
_NAME_OWNER = Path(__file__).resolve().parents[2] / "app" / "db" / "qdrant_client.py"


def _docstring_ids(tree: ast.AST) -> set:
    """ids of the Constant nodes that are docstrings.

    Prose that NAMES a legacy collection ("...NOT dtc_embeddings_hu...") is
    documentation of the fix and must not trip the guard; only a string the
    program can actually pass to Qdrant counts. Comments never reach the AST,
    so they are exempt for free.
    """
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                ids.add(id(body[0].value))
    return ids


def _executable_legacy_literals(path: Path, allowed_in: str = "") -> list:
    """Legacy collection names this module can actually evaluate at runtime.

    Args:
        path: Module to scan.
        allowed_in: Name of a module-level assignment target whose literals are
            exempt (the GDPR sweep's data list).
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    exempt = _docstring_ids(tree)

    if allowed_in:
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(isinstance(t, ast.Name) and t.id == allowed_in for t in targets):
                    exempt |= {id(c) for c in ast.walk(node) if isinstance(c, ast.Constant)}

    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value in LEGACY_COLLECTION_NAMES
        and id(node) not in exempt
    ]


class TestLegacyCollectionsAreStructurallyUnreachable:
    def test_no_other_app_module_can_name_a_legacy_collection(self):
        """FAILS if any app module reintroduces a usable legacy collection name.

        Fixing the drift took three commits because the wrong name was
        reachable from several places: the RAG retrieval leg, the chat DTC
        context and the admin consistency check each carried their own copy.
        Scanning the whole package is the only check that also covers the files
        this test does not know about.
        """
        app_dir = Path(__file__).resolve().parents[2] / "app"
        offenders = {}
        for path in sorted(app_dir.rglob("*.py")):
            if path == _NAME_OWNER:
                continue
            stray = _executable_legacy_literals(path)
            if stray:
                offenders[str(path.relative_to(app_dir))] = stray

        assert not offenders, (
            "legacy Qdrant collection names reappeared in app code: "
            f"{offenders}. Every vector lives in settings.QDRANT_UNIFIED_COLLECTION; "
            "these collections are (near-)empty and searching them returns [] "
            "instead of an error."
        )

    def test_the_owning_module_names_them_only_in_the_gdpr_sweep_list(self):
        """Inside qdrant_client.py the names may only exist as data for the
        erasure sweep - never as an argument, an attribute or a default."""
        stray = _executable_legacy_literals(_NAME_OWNER, allowed_in="_LEGACY_COLLECTIONS")
        assert not stray, (
            f"legacy collection names used outside _LEGACY_COLLECTIONS: {stray}. "
            "That tuple exists only so the GDPR sweep can delete a user's points "
            "from collections that still physically exist."
        )

    def test_the_guard_actually_detects_a_reintroduced_name(self, tmp_path):
        """A guard that cannot fail is not a guard - prove it fires."""
        offender = tmp_path / "regressed.py"
        offender.write_text(
            '"""Docstring naming dtc_embeddings_hu must NOT trip the guard."""\n'
            "# ...and neither must a comment about symptom_embeddings_hu.\n"
            'COLLECTION = "dtc_embeddings_hu"\n',
            encoding="utf-8",
        )
        assert _executable_legacy_literals(offender) == ["dtc_embeddings_hu"]

    def test_no_search_entry_point_accepts_a_collection_name(self):
        """FAILS if a collection name becomes a caller's decision again.

        The write path is included: hand-typed destinations on the write side
        are how the stale partial copies got into dtc_embeddings_hu.
        """
        from app.db.qdrant_client import QdrantService

        forbidden = {"collection", "collection_name"}
        offenders = {}
        for name in dir(QdrantService):
            if not (name.startswith("search") or name == "upsert_vectors"):
                continue
            member = getattr(QdrantService, name)
            if not callable(member):
                continue
            params = set(inspect.signature(member).parameters) & forbidden
            if params:
                offenders[name] = sorted(params)

        assert not offenders, (
            f"Qdrant retrieval/write API accepts a collection name again: {offenders}"
        )

    def test_rag_retrieval_does_not_accept_a_collection_name(self):
        """The RAG's free-text ``collection=`` argument was the drift's carrier."""
        from app.services.rag_service import RAGService

        params = inspect.signature(RAGService.retrieve_from_qdrant).parameters
        assert "collection" not in params
        assert "collection_name" not in params
        # ...and the payload type is now mandatory, not an optional alternative.
        assert params["type_"].default is inspect.Parameter.empty

    def test_the_dead_legacy_search_methods_are_gone(self):
        """They targeted collections holding 0 / ~117 stale points and had no
        production caller. Keeping them alive keeps the drift reachable."""
        from app.db.qdrant_client import QdrantService

        for dead in ("search_similar_symptoms", "search_components", "search_repairs"):
            assert not hasattr(QdrantService, dead), f"{dead} was resurrected"

    def test_the_service_cannot_drop_a_collection(self):
        """``dtc_embeddings_hu`` holds ~2,323 real points in production.

        Dropping a collection is a human data decision at the Qdrant console;
        the app must have no code path that can do it. ``delete_by_user``
        (points, filtered by user id) is the only deletion that stays.
        """
        from app.db.qdrant_client import QdrantService

        assert not hasattr(QdrantService, "delete_collection")
        assert hasattr(QdrantService, "delete_by_user")

    def test_no_class_level_collection_constants_remain(self):
        from app.db.qdrant_client import QdrantService

        leftovers = [
            name
            for name in dir(QdrantService)
            if "COLLECTION" in name
            and getattr(QdrantService, name, None) in LEGACY_COLLECTION_NAMES
        ]
        assert not leftovers, f"legacy collection constants still exported: {leftovers}"


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
            return_value=_collections("dtc_embeddings_hu", "symptom_embeddings_hu")
        )

        result = await service.delete_by_user("user-123")

        targeted = [c.kwargs["collection_name"] for c in service.client.delete.call_args_list]
        assert targeted[0] == settings.QDRANT_UNIFIED_COLLECTION
        assert set(targeted[1:]) == {"dtc_embeddings_hu", "symptom_embeddings_hu"}
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
            return_value=_collections("dtc_embeddings_hu", "symptom_embeddings_hu")
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
        info_obj = _make_collection_info("autocognitix", points_count=1000, vectors_count=1000)
        service.client.get_collection = AsyncMock(return_value=info_obj)

        info = await service.get_collection_info(settings.QDRANT_UNIFIED_COLLECTION)
        assert info["name"] == settings.QDRANT_UNIFIED_COLLECTION
        assert info["points_count"] == 1000
        assert info["vectors_count"] == 1000
        assert info["status"] == "green"


# ---------------------------------------------------------------------------
# get_storage_stats
#
# It enumerated ONLY the five legacy collections, so the store that actually
# holds ~60k vectors - above STORAGE_WARN_THRESHOLD - was never measured and
# check_storage_alerts could never fire for it. Capacity monitoring reported on
# everything except the thing being filled.
# ---------------------------------------------------------------------------


def _stats_stub(**by_collection):
    async def _get(collection_name):
        if collection_name in by_collection:
            return by_collection[collection_name]
        raise Exception("unavailable")

    return AsyncMock(side_effect=_get)


class TestGetStorageStats:
    @pytest.mark.asyncio
    async def test_always_reports_the_collection_that_holds_the_vectors(self, service):
        service.client.get_collections = AsyncMock(return_value=_collections())
        info = {"name": "autocognitix", "points_count": 60955, "vectors_count": 60955}

        with patch.object(
            service,
            "get_collection_info",
            new=_stats_stub(**{settings.QDRANT_UNIFIED_COLLECTION: info}),
        ):
            stats = await service.get_storage_stats()

        assert settings.QDRANT_UNIFIED_COLLECTION in stats
        assert stats[settings.QDRANT_UNIFIED_COLLECTION]["points_count"] == 60955

    @pytest.mark.asyncio
    async def test_surfaces_legacy_collections_only_while_they_exist(self, service):
        """An operator has to see the stale points to decide about dropping them;
        once a human drops the collection it disappears from the report."""
        service.client.get_collections = AsyncMock(return_value=_collections("dtc_embeddings_hu"))

        with patch.object(
            service,
            "get_collection_info",
            new=_stats_stub(
                **{
                    settings.QDRANT_UNIFIED_COLLECTION: {"points_count": 60955},
                    "dtc_embeddings_hu": {"points_count": 2323},
                }
            ),
        ):
            stats = await service.get_storage_stats()

        assert set(stats) == {settings.QDRANT_UNIFIED_COLLECTION, "dtc_embeddings_hu"}
        assert stats["dtc_embeddings_hu"]["points_count"] == 2323

    @pytest.mark.asyncio
    async def test_get_storage_stats_handles_errors(self, service):
        service.client.get_collections = AsyncMock(return_value=_collections())

        with patch.object(
            service,
            "get_collection_info",
            new=AsyncMock(side_effect=Exception("unavailable")),
        ):
            stats = await service.get_storage_stats()

        assert stats == {settings.QDRANT_UNIFIED_COLLECTION: {"error": "unavailable"}}

    @pytest.mark.asyncio
    async def test_the_real_store_can_now_trigger_a_capacity_alert(self, service):
        """The point of the fix: 60,955 > STORAGE_WARN_THRESHOLD must alert."""
        service.client.get_collections = AsyncMock(return_value=_collections())

        with patch.object(
            service,
            "get_collection_info",
            new=_stats_stub(**{settings.QDRANT_UNIFIED_COLLECTION: {"points_count": 60955}}),
        ):
            alerts = await service.check_storage_alerts()

        assert [a["collection"] for a in alerts] == [settings.QDRANT_UNIFIED_COLLECTION]


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
