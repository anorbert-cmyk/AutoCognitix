"""
API-level proof that a broken embedding backend DEGRADES instead of 500-ing.

Making the embedding path fail loudly (``EmbeddingUnavailableError`` instead of
a silent ``[0.0] * 768``) is only safe if every endpoint that touches semantic
search absorbs it. Otherwise "fail loudly" would turn a quiet quality problem
into a production outage.

These tests hit the real endpoint through the real router, with only the
embedding seam replaced.
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient

from app.core.exceptions import EmbeddingUnavailableError


def _exploding_embedding_service() -> MagicMock:
    """An embedding service whose async embed raises EmbeddingUnavailableError."""
    stub = MagicMock()
    stub.embed_text_async = AsyncMock(side_effect=EmbeddingUnavailableError())
    return stub


class TestDTCSearchDegradesGracefully:
    @pytest.mark.asyncio
    async def test_semantic_search_still_returns_200_when_embedding_is_unavailable(
        self, async_client: AsyncClient, sample_dtc_codes
    ):
        """The established pattern: log at ERROR, fall back to lexical results."""
        with patch(
            "app.api.v1.endpoints.dtc_codes.get_embedding_service",
            return_value=_exploding_embedding_service(),
        ):
            response = await async_client.get(
                "/api/v1/dtc/search",
                params={"q": "Mass Air Flow", "use_semantic": "true", "skip_cache": "true"},
            )

        assert response.status_code == 200
        # Lexical results survive; only the semantic leg is lost.
        assert isinstance(response.json(), list)

    @pytest.mark.asyncio
    async def test_semantic_search_still_returns_200_when_qdrant_rejects_the_vector(
        self, async_client: AsyncClient, sample_dtc_codes
    ):
        """The Qdrant norm guard raises ValueError; that must not escape either."""
        embedding_stub = MagicMock()
        embedding_stub.embed_text_async = AsyncMock(return_value=[0.0] * 768)

        qdrant_stub = MagicMock()
        qdrant_stub.search_dtc = AsyncMock(
            side_effect=ValueError("Degenerate query vector (norm=0.0)")
        )

        with (
            patch(
                "app.api.v1.endpoints.dtc_codes.get_embedding_service",
                return_value=embedding_stub,
            ),
            patch("app.api.v1.endpoints.dtc_codes.qdrant_client", qdrant_stub),
        ):
            response = await async_client.get(
                "/api/v1/dtc/search",
                params={"q": "Levegotomeg", "use_semantic": "true", "skip_cache": "true"},
            )

        assert response.status_code == 200
        assert isinstance(response.json(), list)

    @pytest.mark.asyncio
    async def test_lexical_only_search_never_touches_the_embedding_backend(
        self, async_client: AsyncClient, sample_dtc_codes
    ):
        """use_semantic=false must not even construct the embedding service."""
        stub = _exploding_embedding_service()
        with patch(
            "app.api.v1.endpoints.dtc_codes.get_embedding_service", return_value=stub
        ) as get_svc:
            response = await async_client.get(
                "/api/v1/dtc/search",
                params={"q": "Mass Air Flow", "use_semantic": "false", "skip_cache": "true"},
            )

        assert response.status_code == 200
        get_svc.assert_not_called()


class TestDegradationIsVisibleToOnCall:
    """Degrading quietly is the bug, not the fix.

    Sentry's logging integration (``app/core/logging.py``) only raises events
    from ``logging.ERROR`` upwards. Both call sites that ABSORB
    ``EmbeddingUnavailableError`` must therefore log at ERROR - otherwise a dead
    backend degrades every semantic search to lexical while on-call sees
    nothing, which is precisely the operator experience of the silent zero
    vector this whole change exists to eliminate.
    """

    _DTC_LOGGER = "app.api.v1.endpoints.dtc_codes"
    _RAG_LOGGER = "app.services.rag_service"

    @pytest.mark.asyncio
    async def test_dtc_search_logs_a_dead_backend_at_error(
        self, async_client: AsyncClient, sample_dtc_codes, caplog
    ):
        with (
            caplog.at_level(logging.WARNING, logger=self._DTC_LOGGER),
            patch(
                "app.api.v1.endpoints.dtc_codes.get_embedding_service",
                return_value=_exploding_embedding_service(),
            ),
        ):
            response = await async_client.get(
                "/api/v1/dtc/search",
                params={"q": "Mass Air Flow", "use_semantic": "true", "skip_cache": "true"},
            )

        assert response.status_code == 200
        records = [
            r
            for r in caplog.records
            if r.name == self._DTC_LOGGER and "Embedding backend unavailable" in r.getMessage()
        ]
        assert records, "an embedding outage produced no log record at all"
        assert all(r.levelno == logging.ERROR for r in records), (
            "logged below ERROR - Sentry would never see it"
        )
        assert all(r.exc_info for r in records), "no traceback attached"

    @pytest.mark.asyncio
    async def test_a_transient_qdrant_failure_stays_at_warning(
        self, async_client: AsyncClient, sample_dtc_codes, caplog
    ):
        """Only the dead-backend case is ERROR. If everything were bumped to
        ERROR the new signal would drown in the old noise."""
        embedding_stub = MagicMock()
        embedding_stub.embed_text_async = AsyncMock(return_value=[0.1] * 768)

        qdrant_stub = MagicMock()
        qdrant_stub.search_dtc = AsyncMock(side_effect=RuntimeError("qdrant timeout"))

        with (
            caplog.at_level(logging.WARNING, logger=self._DTC_LOGGER),
            patch(
                "app.api.v1.endpoints.dtc_codes.get_embedding_service",
                return_value=embedding_stub,
            ),
            patch("app.api.v1.endpoints.dtc_codes.qdrant_client", qdrant_stub),
        ):
            response = await async_client.get(
                "/api/v1/dtc/search",
                params={"q": "Levegotomeg aramlas", "use_semantic": "true", "skip_cache": "true"},
            )

        assert response.status_code == 200
        records = [
            r
            for r in caplog.records
            if r.name == self._DTC_LOGGER and "Semantic search failed" in r.getMessage()
        ]
        assert records, "the Qdrant failure produced no log record at all"
        assert all(r.levelno == logging.WARNING for r in records)

    @pytest.mark.asyncio
    async def test_rag_retrieval_logs_a_dead_backend_at_error(self, caplog):
        """The RAG retrieval leg is the other absorber - same rule applies."""
        from app.services.rag_service import RAGService

        service = RAGService()

        with (
            caplog.at_level(logging.WARNING, logger=self._RAG_LOGGER),
            patch(
                "app.services.rag_service.embed_text_async",
                AsyncMock(side_effect=EmbeddingUnavailableError()),
            ),
        ):
            items = await service.retrieve_from_qdrant(
                "nincs elerheto embedding backend", type_="dtc"
            )

        assert items == []
        records = [
            r
            for r in caplog.records
            if r.name == self._RAG_LOGGER and "Embedding backend unavailable" in r.getMessage()
        ]
        assert records, "an embedding outage produced no log record at all"
        assert all(r.levelno == logging.ERROR for r in records)
        assert all(r.exc_info for r in records), "no traceback attached"
        # The old message promised a keyword fallback that does not exist: this
        # branch returns [] and the caller loses the semantic leg entirely.
        assert all("keyword search" not in r.getMessage() for r in records)
