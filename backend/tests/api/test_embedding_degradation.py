"""
API-level proof that a broken embedding backend DEGRADES instead of 500-ing.

Making the embedding path fail loudly (``EmbeddingUnavailableError`` instead of
a silent ``[0.0] * 768``) is only safe if every endpoint that touches semantic
search absorbs it. Otherwise "fail loudly" would turn a quiet quality problem
into a production outage.

These tests hit the real endpoint through the real router, with only the
embedding seam replaced.
"""

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
