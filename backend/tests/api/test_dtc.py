"""
API tests for DTC (Diagnostic Trouble Code) endpoints.

Tests:
- GET /api/v1/dtc/search - Search DTC codes by code or description
- GET /api/v1/dtc/categories/list - Get DTC categories
- GET /api/v1/dtc/{code} - Get DTC code details
- GET /api/v1/dtc/{code}/related - Get related DTC codes
- POST /api/v1/dtc/ - Create new DTC code
- POST /api/v1/dtc/bulk - Bulk import DTC codes
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.postgres.models import DTCCode
from app.db.qdrant_client import QdrantService


def _qdrant_with_mocked_search(hits: list) -> QdrantService:
    """Build a real QdrantService whose only mocked seam is the low-level
    ``search``.

    This exercises the true ``search_dtc`` -> ``search_unified`` -> ``search``
    routing (so the collection/type fix is under test) while returning canned
    ``autocognitix``-style hits and opening no network connection.
    """
    service = QdrantService.__new__(QdrantService)  # bypass __init__ (no network)
    service.search = AsyncMock(return_value=hits)  # type: ignore[method-assign]
    return service


def _embedding_service_stub() -> MagicMock:
    """Stub embedding service returning a fixed 768-dim query vector."""
    stub = MagicMock()
    stub.embed_text_async = AsyncMock(return_value=[0.0] * 768)
    return stub


class TestDTCSearch:
    """Tests for GET /api/v1/dtc/search endpoint."""

    @pytest.mark.asyncio
    async def test_search_by_code_returns_200(self, async_client: AsyncClient, sample_dtc_codes):
        """Test searching by DTC code returns 200."""
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "P0101"},
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_search_by_code_returns_matching_results(
        self, async_client: AsyncClient, sample_dtc_codes
    ):
        """Test searching by DTC code returns matching results."""
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "P0101"},
        )

        assert response.status_code == 200
        data = response.json()

        # Should find P0101
        codes = [d["code"] for d in data]
        assert "P0101" in codes

    @pytest.mark.asyncio
    async def test_search_by_description_text(self, async_client: AsyncClient, sample_dtc_codes):
        """Test searching by description text."""
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "Mass Air Flow", "use_semantic": "false"},
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_search_by_hungarian_description(
        self, async_client: AsyncClient, sample_dtc_codes
    ):
        """Test searching by Hungarian description."""
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "Levegotomeg", "use_semantic": "false"},
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_search_with_category_filter(self, async_client: AsyncClient, sample_dtc_codes):
        """Test searching with category filter."""
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "P", "category": "powertrain"},
        )

        assert response.status_code == 200
        data = response.json()

        # All results should be powertrain category
        for dtc in data:
            if "category" in dtc:
                assert dtc["category"] == "powertrain"

    @pytest.mark.asyncio
    async def test_search_with_limit(self, async_client: AsyncClient, sample_dtc_codes):
        """Test search respects limit parameter."""
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "P", "limit": 2},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) <= 2

    @pytest.mark.asyncio
    async def test_search_empty_query_returns_422(self, async_client: AsyncClient):
        """Test that empty query returns 422."""
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": ""},
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_search_missing_query_returns_422(self, async_client: AsyncClient):
        """Test that missing query parameter returns 422."""
        response = await async_client.get("/api/v1/dtc/search")

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_search_results_include_relevance_score(
        self, async_client: AsyncClient, sample_dtc_codes
    ):
        """Test that search results include relevance score."""
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "P0101"},
        )

        assert response.status_code == 200
        data = response.json()

        if data:
            # First result should have relevance_score
            assert "relevance_score" in data[0]
            assert isinstance(data[0]["relevance_score"], (int, float))

    @pytest.mark.asyncio
    async def test_search_results_sorted_by_relevance(
        self, async_client: AsyncClient, sample_dtc_codes
    ):
        """Test that search results are sorted by relevance."""
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "P0101"},
        )

        assert response.status_code == 200
        data = response.json()

        if len(data) > 1:
            # Results should be sorted by relevance (descending)
            scores = [d.get("relevance_score", 0) for d in data]
            assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_search_partial_code_match(self, async_client: AsyncClient, sample_dtc_codes):
        """Test searching with partial DTC code prefix."""
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "P01"},
        )

        assert response.status_code == 200
        data = response.json()

        # Should find codes starting with P01
        for dtc in data:
            assert dtc["code"].startswith("P0") or "01" in dtc["code"]

    @pytest.mark.asyncio
    async def test_search_case_insensitive(self, async_client: AsyncClient, sample_dtc_codes):
        """Test that search is case insensitive."""
        response_upper = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "P0101"},
        )
        response_lower = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "p0101"},
        )

        assert response_upper.status_code == 200
        assert response_lower.status_code == 200

        data_upper = response_upper.json()
        data_lower = response_lower.json()

        # Both should return same codes
        codes_upper = {d["code"] for d in data_upper}
        codes_lower = {d["code"] for d in data_lower}
        assert codes_upper == codes_lower

    @pytest.mark.asyncio
    async def test_search_with_skip_cache(self, async_client: AsyncClient, sample_dtc_codes):
        """Test search with skip_cache parameter."""
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "P0101", "skip_cache": "true"},
        )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_semantic_search_enriches_unified_hits_from_postgres(
        self, async_client: AsyncClient, sample_dtc_codes
    ):
        """REGRESSION (Qdrant collection drift): a Hungarian symptom query returns
        real, PostgreSQL-enriched results and preserves the Qdrant relevance score.

        The unified ``autocognitix`` DTC payload carries only ``{type, code,
        description, category, subcategory}`` — so ``description_hu`` in the
        response can ONLY come from the PostgreSQL enrichment, proving the
        semantic hit was mapped to a full DTCSearchResult.
        """
        autocognitix_hit = {
            "id": 1,
            "score": 0.82,
            "payload": {
                "type": "dtc",
                "code": "P0101",
                "description": "Mass Air Flow Circuit Range/Performance",
                "category": "powertrain",
                "subcategory": "",
            },
        }
        qdrant = _qdrant_with_mocked_search([autocognitix_hit])

        with (
            patch("app.api.v1.endpoints.dtc_codes.qdrant_client", qdrant),
            patch(
                "app.api.v1.endpoints.dtc_codes.get_embedding_service",
                return_value=_embedding_service_stub(),
            ),
        ):
            response = await async_client.get(
                "/api/v1/dtc/search",
                params={"q": "motor rángatás", "skip_cache": "true"},
            )

        assert response.status_code == 200
        data = response.json()

        codes = [d["code"] for d in data]
        assert "P0101" in codes

        p0101 = next(d for d in data if d["code"] == "P0101")
        # Enrichment proof: description_hu is NOT in the Qdrant payload.
        assert p0101["description_hu"] == "Levegotomeg-mero aramkor tartomany/teljesitmeny hiba"
        assert p0101["severity"] == "medium"
        # Qdrant relevance score is preserved on the enriched result.
        assert p0101["relevance_score"] == pytest.approx(0.82)

        # The mocked seam confirms we hit the unified collection with type=dtc.
        # There is no collection_name kwarg any more: QdrantService.search
        # resolves settings.QDRANT_UNIFIED_COLLECTION itself, so a caller cannot
        # aim this at a legacy collection.
        _, kwargs = qdrant.search.call_args
        assert "collection_name" not in kwargs
        assert kwargs["filter_conditions"]["type"] == "dtc"

    @pytest.mark.asyncio
    async def test_semantic_search_skips_codes_missing_in_postgres(
        self, async_client: AsyncClient, sample_dtc_codes
    ):
        """Codes returned by Qdrant but absent from PostgreSQL are skipped, not
        crashed on."""
        hits = [
            {
                "id": 1,
                "score": 0.95,
                "payload": {"type": "dtc", "code": "P9999", "category": "powertrain"},
            },
            {
                "id": 2,
                "score": 0.80,
                "payload": {"type": "dtc", "code": "P0101", "category": "powertrain"},
            },
        ]
        qdrant = _qdrant_with_mocked_search(hits)

        with (
            patch("app.api.v1.endpoints.dtc_codes.qdrant_client", qdrant),
            patch(
                "app.api.v1.endpoints.dtc_codes.get_embedding_service",
                return_value=_embedding_service_stub(),
            ),
        ):
            response = await async_client.get(
                "/api/v1/dtc/search",
                params={"q": "füst a kipufogóból", "skip_cache": "true"},
            )

        assert response.status_code == 200
        codes = [d["code"] for d in response.json()]
        assert "P0101" in codes  # exists in PostgreSQL -> enriched
        assert "P9999" not in codes  # missing in PostgreSQL -> skipped

    @pytest.mark.asyncio
    async def test_semantic_failure_falls_back_to_lexical_no_500(
        self, async_client: AsyncClient, sample_dtc_codes
    ):
        """A Qdrant failure must not 500 the endpoint; it falls back to lexical
        results already collected from PostgreSQL."""
        qdrant = QdrantService.__new__(QdrantService)
        qdrant.search = AsyncMock(side_effect=RuntimeError("Qdrant down"))  # type: ignore[method-assign]

        with (
            patch("app.api.v1.endpoints.dtc_codes.qdrant_client", qdrant),
            patch(
                "app.api.v1.endpoints.dtc_codes.get_embedding_service",
                return_value=_embedding_service_stub(),
            ),
        ):
            # "Levegotomeg" matches P0101's Hungarian description via the lexical
            # path; the semantic path runs (and fails) but must be swallowed.
            response = await async_client.get(
                "/api/v1/dtc/search",
                params={"q": "Levegotomeg", "skip_cache": "true"},
            )

        assert response.status_code == 200
        codes = [d["code"] for d in response.json()]
        assert "P0101" in codes

    @pytest.mark.asyncio
    async def test_semantic_search_clamps_out_of_range_scores(
        self, async_client: AsyncClient, sample_dtc_codes
    ):
        """Raw cosine similarity is in [-1, 1], but DTCSearchResult.relevance_score
        is bounded to [0, 1]. An unclamped out-of-range hit would raise
        ValidationError, get swallowed by the except, and silently degrade to a
        lexical-only response (the drift bug reappearing). Both a negative and a
        >1 hit must survive, clamped into range.
        """
        hits = [
            {
                "id": 1,
                "score": -0.05,  # below the 0 lower bound
                "payload": {"type": "dtc", "code": "P0101", "category": "powertrain"},
            },
            {
                "id": 2,
                "score": 1.0000001,  # just above the 1 upper bound (float noise)
                "payload": {"type": "dtc", "code": "P0171", "category": "powertrain"},
            },
        ]
        qdrant = _qdrant_with_mocked_search(hits)

        with (
            patch("app.api.v1.endpoints.dtc_codes.qdrant_client", qdrant),
            patch(
                "app.api.v1.endpoints.dtc_codes.get_embedding_service",
                return_value=_embedding_service_stub(),
            ),
        ):
            response = await async_client.get(
                "/api/v1/dtc/search",
                params={"q": "motor rángatás", "skip_cache": "true"},
            )

        assert response.status_code == 200
        data = response.json()
        by_code = {d["code"]: d for d in data}

        # Both codes survived — no ValidationError swallowed into a lexical-only fallback.
        assert "P0101" in by_code
        assert "P0171" in by_code

        # Scores are clamped into the [0, 1] bound the schema requires.
        assert by_code["P0101"]["relevance_score"] == 0.0
        assert by_code["P0171"]["relevance_score"] == 1.0
        for d in data:
            assert 0.0 <= d["relevance_score"] <= 1.0


class TestDTCSearchCodeVsFreeTextRouting:
    """The "is this query a code?" branch that gates the semantic search.

    A true verdict takes the exact-match shortcut and skips the embedding call;
    a false verdict is what routes Hungarian free text to semantic search. The
    old test accepted any all-caps string starting with P/B/C/U, so a mechanic
    typing in capitals silently lost semantic search entirely.
    """

    async def _search(self, async_client: AsyncClient, query: str):
        qdrant = _qdrant_with_mocked_search([])
        embedding = _embedding_service_stub()
        with (
            patch("app.api.v1.endpoints.dtc_codes.qdrant_client", qdrant),
            patch(
                "app.api.v1.endpoints.dtc_codes.get_embedding_service",
                return_value=embedding,
            ),
        ):
            response = await async_client.get(
                "/api/v1/dtc/search",
                params={"q": query, "use_semantic": "true", "skip_cache": "true"},
            )
        assert response.status_code == 200, response.text
        return embedding

    @pytest.mark.asyncio
    @pytest.mark.parametrize("query", ["P", "P0", "P03", "P030", "P0300", "P26B7", "p26b7"])
    async def test_code_and_partial_code_skip_semantic_search(
        self, async_client: AsyncClient, sample_dtc_codes, query: str
    ):
        """A complete code, and a code still being typed, stay on the lexical path.

        The autocomplete fires from two characters, so narrowing this must not
        start an embedding round-trip on every keystroke.
        """
        embedding = await self._search(async_client, query)
        embedding.embed_text_async.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("query", ["PORLASZTO", "PEACE", "BUXA HANG", "CSAPAGY", "P9324"])
    async def test_capitalised_free_text_still_reaches_semantic_search(
        self, async_client: AsyncClient, sample_dtc_codes, query: str
    ):
        """Free text in capitals is not a code and must keep semantic search."""
        embedding = await self._search(async_client, query)
        embedding.embed_text_async.assert_called_once()


class TestDTCSearchHasNoScoreThreshold:
    """The semantic leg of ``/dtc/search`` filters by NOTHING but rank.

    This pins a fact that is easy to assume away: ``search_dtc`` never passes a
    ``score_threshold``, so ``QdrantService.search`` never sets the key and
    Qdrant applies no score cutoff at all. The only threshold in the codebase
    (``rag_service.retrieve_from_qdrant``, default 0.5) belongs to
    ``/diagnosis/analyze`` and is not on this path.

    Why pin it: when Hungarian symptom queries come back empty, "the similarity
    threshold is too strict" is the intuitive diagnosis and the wrong one -
    there is no threshold to loosen, so loosening one cannot add a single
    result. If a threshold is ever introduced here, these tests fail and force
    that decision to be argued explicitly with measured precision numbers
    rather than slipped in as a fix for empty results.
    """

    @pytest.mark.asyncio
    async def test_no_score_threshold_is_sent_to_qdrant(
        self, async_client: AsyncClient, sample_dtc_codes
    ):
        """The whole call chain leaves ``score_threshold`` unset."""
        qdrant = _qdrant_with_mocked_search([])

        with (
            patch("app.api.v1.endpoints.dtc_codes.qdrant_client", qdrant),
            patch(
                "app.api.v1.endpoints.dtc_codes.get_embedding_service",
                return_value=_embedding_service_stub(),
            ),
        ):
            response = await async_client.get(
                "/api/v1/dtc/search",
                params={"q": "motor rángatás", "skip_cache": "true"},
            )

        assert response.status_code == 200
        _, kwargs = qdrant.search.call_args
        assert kwargs.get("score_threshold") is None, (
            "a score cutoff appeared on the DTC search path; empty Hungarian "
            "results are not caused by scoring, so this needs its own justification"
        )

    @pytest.mark.asyncio
    async def test_a_near_zero_scoring_hit_still_reaches_the_client(
        self, async_client: AsyncClient, sample_dtc_codes
    ):
        """Even a 0.02-similarity hit is served, so recall is not score-limited.

        ``motor rángatás`` matches nothing lexically, so every row in the
        response had to come from the semantic leg.
        """
        hits = [
            {
                "id": 1,
                "score": 0.02,  # far below any plausible cutoff
                "payload": {"type": "dtc", "code": "P0101", "category": "powertrain"},
            }
        ]
        qdrant = _qdrant_with_mocked_search(hits)

        with (
            patch("app.api.v1.endpoints.dtc_codes.qdrant_client", qdrant),
            patch(
                "app.api.v1.endpoints.dtc_codes.get_embedding_service",
                return_value=_embedding_service_stub(),
            ),
        ):
            response = await async_client.get(
                "/api/v1/dtc/search",
                params={"q": "motor rángatás", "skip_cache": "true"},
            )

        assert response.status_code == 200
        data = response.json()
        assert [d["code"] for d in data] == ["P0101"]
        assert data[0]["relevance_score"] == pytest.approx(0.02)

    @pytest.mark.asyncio
    async def test_hungarian_substring_hits_are_lexical_not_semantic(
        self, async_client: AsyncClient, sample_dtc_codes
    ):
        """The 0.7 score users see is a hardcoded ILIKE constant, not a cosine.

        Production returns ``gyújtáskimaradás`` at exactly 0.7 because the term
        appears verbatim inside ``description_hu``; ``_compute_text_relevance``
        stamps that constant on every ``description_hu`` substring match. It
        looks like working semantic search and is not: with Qdrant returning
        nothing at all, the identical rows and identical scores come back.
        """
        qdrant = _qdrant_with_mocked_search([])  # semantic leg contributes nothing

        with (
            patch("app.api.v1.endpoints.dtc_codes.qdrant_client", qdrant),
            patch(
                "app.api.v1.endpoints.dtc_codes.get_embedding_service",
                return_value=_embedding_service_stub(),
            ),
        ):
            response = await async_client.get(
                "/api/v1/dtc/search",
                params={"q": "Levegotomeg", "skip_cache": "true"},
            )

        assert response.status_code == 200
        data = response.json()
        by_code = {d["code"]: d for d in data}
        assert "P0101" in by_code
        assert by_code["P0101"]["relevance_score"] == pytest.approx(0.7)


class TestDTCSearchSemanticIsGatedByLimit:
    """``len(results) < limit`` decides whether semantic search runs at all.

    A CHARACTERIZATION test: it documents current behaviour rather than
    endorsing it. The consequence is real - a page already filled by lexical
    substring matches never gets a semantic hit, no matter how much better that
    hit would rank. Because lexical scores cap at 0.7 while a good cosine hit
    scores higher, the sort at the end of the handler cannot repair it: the
    better rows were never fetched. It bites hardest at the small limits an
    autocomplete uses, which is why one constant cannot serve both a one-word
    query and a five-word symptom sentence.

    Left unchanged deliberately: with the semantic leg returning nothing in
    production, opening this gate buys no measurable recall today and costs an
    embedding round-trip on every free-text search. Revisit once the DTC
    vectors are retrievable again.
    """

    @pytest.mark.asyncio
    async def test_a_full_page_of_lexical_hits_suppresses_semantic_search(
        self, async_client: AsyncClient, sample_dtc_codes
    ):
        """limit=1 filled lexically -> the embedding backend is never touched."""
        embedding = _embedding_service_stub()

        with (
            patch("app.api.v1.endpoints.dtc_codes.qdrant_client", _qdrant_with_mocked_search([])),
            patch(
                "app.api.v1.endpoints.dtc_codes.get_embedding_service",
                return_value=embedding,
            ),
        ):
            response = await async_client.get(
                "/api/v1/dtc/search",
                params={"q": "Mass Air Flow", "limit": 1, "skip_cache": "true"},
            )

        assert response.status_code == 200
        assert len(response.json()) == 1
        embedding.embed_text_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_the_same_query_with_room_left_does_run_semantic_search(
        self, async_client: AsyncClient, sample_dtc_codes
    ):
        """Same query, larger limit -> the gate opens. Only ``limit`` changed."""
        embedding = _embedding_service_stub()

        with (
            patch("app.api.v1.endpoints.dtc_codes.qdrant_client", _qdrant_with_mocked_search([])),
            patch(
                "app.api.v1.endpoints.dtc_codes.get_embedding_service",
                return_value=embedding,
            ),
        ):
            response = await async_client.get(
                "/api/v1/dtc/search",
                params={"q": "Mass Air Flow", "limit": 20, "skip_cache": "true"},
            )

        assert response.status_code == 200
        embedding.embed_text_async.assert_called_once()


class TestDTCCategories:
    """Tests for GET /api/v1/dtc/categories/list endpoint."""

    @pytest.mark.asyncio
    async def test_get_categories_returns_200(self, async_client: AsyncClient):
        """Test getting categories returns 200."""
        response = await async_client.get("/api/v1/dtc/categories/list")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_get_categories_returns_all_four(self, async_client: AsyncClient):
        """Test that all four DTC categories are returned."""
        response = await async_client.get("/api/v1/dtc/categories/list")

        assert response.status_code == 200
        data = response.json()

        codes = [c["code"] for c in data]
        assert "P" in codes  # Powertrain
        assert "B" in codes  # Body
        assert "C" in codes  # Chassis
        assert "U" in codes  # Network

    @pytest.mark.asyncio
    async def test_categories_include_hungarian_names(self, async_client: AsyncClient):
        """Test that categories include Hungarian names."""
        response = await async_client.get("/api/v1/dtc/categories/list")

        assert response.status_code == 200
        data = response.json()

        for category in data:
            assert "name_hu" in category
            assert category["name_hu"]  # Not empty

    @pytest.mark.asyncio
    async def test_categories_include_descriptions(self, async_client: AsyncClient):
        """Test that categories include descriptions."""
        response = await async_client.get("/api/v1/dtc/categories/list")

        assert response.status_code == 200
        data = response.json()

        for category in data:
            assert "description" in category
            assert "description_hu" in category


class TestDTCCodeDetail:
    """Tests for GET /api/v1/dtc/{code} endpoint."""

    @pytest.mark.asyncio
    async def test_get_dtc_detail_returns_200(self, async_client: AsyncClient, sample_dtc_codes):
        """Test getting DTC detail returns 200."""
        response = await async_client.get("/api/v1/dtc/P0101")

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_get_dtc_detail_returns_full_data(
        self, async_client: AsyncClient, sample_dtc_codes
    ):
        """Test getting DTC detail returns full data."""
        response = await async_client.get("/api/v1/dtc/P0101")

        assert response.status_code == 200
        data = response.json()

        assert data["code"] == "P0101"
        assert "description_en" in data
        assert "description_hu" in data
        assert "category" in data
        assert "severity" in data
        assert "symptoms" in data
        assert "possible_causes" in data
        assert "diagnostic_steps" in data

    @pytest.mark.asyncio
    async def test_get_dtc_detail_nonexistent_returns_404(
        self, async_client: AsyncClient, sample_dtc_codes
    ):
        """Test getting nonexistent DTC returns 404."""
        # Structurally valid (P + 3 + FFF) but not seeded, so the 404 comes from
        # the database lookup rather than from format validation. "P9999", the
        # previous probe, is not a DTC under SAE J2012 (second character 9).
        response = await async_client.get("/api/v1/dtc/P3FFF")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_dtc_detail_invalid_format_returns_400(self, async_client: AsyncClient):
        """Test getting DTC with invalid format returns 400."""
        response = await async_client.get("/api/v1/dtc/INVALID")

        assert response.status_code == 400

    @pytest.mark.asyncio
    @pytest.mark.parametrize("code", ["P26B7", "P090C", "P0A94", "B00A0"])
    async def test_get_dtc_detail_accepts_hex_code_format(
        self, async_client: AsyncClient, sample_dtc_codes, code: str
    ):
        """Real hex DTCs must pass format validation (404, never 400)."""
        response = await async_client.get(f"/api/v1/dtc/{code}")

        assert response.status_code == 404, response.text

    @pytest.mark.asyncio
    @pytest.mark.parametrize("junk", ["PEACE", "P9324", "P9999", "UA80E", "PEACEFUL"])
    async def test_get_dtc_detail_rejects_junk_before_lookup(
        self, async_client: AsyncClient, junk: str
    ):
        """Junk that merely starts with P/B/C/U must not reach Neo4j or the cache."""
        response = await async_client.get(f"/api/v1/dtc/{junk}")

        assert response.status_code == 400, response.text

    @pytest.mark.asyncio
    async def test_get_dtc_detail_case_insensitive(
        self, async_client: AsyncClient, sample_dtc_codes
    ):
        """Test that DTC lookup is case insensitive."""
        response = await async_client.get("/api/v1/dtc/p0101")

        assert response.status_code == 200
        data = response.json()
        assert data["code"] == "P0101"

    @pytest.mark.asyncio
    async def test_get_dtc_detail_with_include_graph_false(
        self, async_client: AsyncClient, sample_dtc_codes
    ):
        """Test getting DTC detail without graph data."""
        response = await async_client.get(
            "/api/v1/dtc/P0101",
            params={"include_graph": "false"},
        )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_get_dtc_detail_with_skip_cache(
        self, async_client: AsyncClient, sample_dtc_codes
    ):
        """Test getting DTC detail with skip_cache."""
        response = await async_client.get(
            "/api/v1/dtc/P0101",
            params={"skip_cache": "true"},
        )

        assert response.status_code == 200


class TestDTCRelatedCodes:
    """Tests for GET /api/v1/dtc/{code}/related endpoint."""

    @pytest.mark.asyncio
    async def test_get_related_codes_returns_200(self, async_client: AsyncClient, sample_dtc_codes):
        """Test getting related codes returns 200."""
        response = await async_client.get("/api/v1/dtc/P0101/related")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_get_related_codes_excludes_original(
        self, async_client: AsyncClient, sample_dtc_codes
    ):
        """Test that related codes exclude the original code."""
        response = await async_client.get("/api/v1/dtc/P0101/related")

        assert response.status_code == 200
        data = response.json()

        codes = [d["code"] for d in data]
        assert "P0101" not in codes

    @pytest.mark.asyncio
    async def test_get_related_codes_with_limit(self, async_client: AsyncClient, sample_dtc_codes):
        """Test getting related codes respects limit."""
        response = await async_client.get(
            "/api/v1/dtc/P0101/related",
            params={"limit": 3},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) <= 3

    @pytest.mark.asyncio
    async def test_get_related_codes_nonexistent_returns_404(
        self, async_client: AsyncClient, sample_dtc_codes
    ):
        """Test getting related codes for nonexistent DTC returns 404."""
        response = await async_client.get("/api/v1/dtc/P3FFF/related")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_related_codes_include_relevance_score(
        self, async_client: AsyncClient, sample_dtc_codes
    ):
        """Test that related codes include relevance score."""
        response = await async_client.get("/api/v1/dtc/P0101/related")

        assert response.status_code == 200
        data = response.json()

        if data:
            for dtc in data:
                assert "relevance_score" in dtc


class TestDTCCreate:
    """Tests for POST /api/v1/dtc/ endpoint."""

    @pytest.mark.asyncio
    async def test_create_dtc_returns_201(
        self, async_client: AsyncClient, dtc_create_data: dict, admin_auth_headers: dict
    ):
        """Test creating DTC returns 201."""
        response = await async_client.post(
            "/api/v1/dtc/",
            json=dtc_create_data,
            headers=admin_auth_headers,
        )

        assert response.status_code == 201
        data = response.json()
        assert data["code"] == dtc_create_data["code"].upper()

    @pytest.mark.asyncio
    async def test_create_dtc_normalizes_code_to_uppercase(
        self, async_client: AsyncClient, admin_auth_headers: dict
    ):
        """Test that created DTC code is normalized to uppercase."""
        response = await async_client.post(
            "/api/v1/dtc/",
            json={
                "code": "p3888",  # lowercase
                "description_en": "Test code",
                "category": "powertrain",
                "severity": "medium",
            },
            headers=admin_auth_headers,
        )

        assert response.status_code == 201
        data = response.json()
        assert data["code"] == "P3888"

    @pytest.mark.asyncio
    async def test_create_duplicate_dtc_returns_400(
        self, async_client: AsyncClient, sample_dtc_codes, admin_auth_headers: dict
    ):
        """Test creating duplicate DTC returns 400."""
        response = await async_client.post(
            "/api/v1/dtc/",
            json={
                "code": "P0101",  # Already exists
                "description_en": "Duplicate",
                "category": "powertrain",
                "severity": "medium",
            },
            headers=admin_auth_headers,
        )

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_create_dtc_missing_required_fields_returns_422(
        self, async_client: AsyncClient, admin_auth_headers: dict
    ):
        """Test creating DTC without required fields returns 422."""
        response = await async_client.post(
            "/api/v1/dtc/",
            json={"code": "P9999"},  # Missing description_en, category, severity
            headers=admin_auth_headers,
        )

        assert response.status_code == 422


class TestDTCBulkImport:
    """Tests for POST /api/v1/dtc/bulk endpoint."""

    @pytest.mark.asyncio
    async def test_bulk_import_returns_201(
        self, async_client: AsyncClient, admin_auth_headers: dict
    ):
        """Test bulk import returns 201."""
        response = await async_client.post(
            "/api/v1/dtc/bulk",
            headers=admin_auth_headers,
            json={
                "codes": [
                    {
                        "code": "P3777",
                        "description_en": "Bulk test 1",
                        "category": "powertrain",
                        "severity": "low",
                    },
                    {
                        "code": "P3778",
                        "description_en": "Bulk test 2",
                        "category": "powertrain",
                        "severity": "medium",
                    },
                ],
                "overwrite_existing": False,
            },
        )

        assert response.status_code in (200, 201)
        data = response.json()

        assert "created" in data
        assert "updated" in data
        assert "skipped" in data
        assert "total" in data

    @pytest.mark.asyncio
    async def test_bulk_import_counts_created(
        self, async_client: AsyncClient, admin_auth_headers: dict
    ):
        """Test that bulk import correctly counts created codes."""
        response = await async_client.post(
            "/api/v1/dtc/bulk",
            headers=admin_auth_headers,
            json={
                "codes": [
                    {
                        "code": "P3666",
                        "description_en": "New code 1",
                        "category": "powertrain",
                        "severity": "low",
                    },
                    {
                        "code": "P3667",
                        "description_en": "New code 2",
                        "category": "powertrain",
                        "severity": "low",
                    },
                ],
                "overwrite_existing": False,
            },
        )

        assert response.status_code in (200, 201)
        data = response.json()

        assert data["created"] == 2
        assert data["total"] == 2

    @pytest.mark.asyncio
    async def test_bulk_import_skips_existing_without_overwrite(
        self, async_client: AsyncClient, sample_dtc_codes, admin_auth_headers: dict
    ):
        """Test that bulk import skips existing codes when overwrite=false."""
        response = await async_client.post(
            "/api/v1/dtc/bulk",
            headers=admin_auth_headers,
            json={
                "codes": [
                    {
                        "code": "P0101",  # Already exists
                        "description_en": "Updated description",
                        "category": "powertrain",
                        "severity": "high",
                    },
                ],
                "overwrite_existing": False,
            },
        )

        assert response.status_code in (200, 201)
        data = response.json()

        assert data["skipped"] == 1
        assert data["created"] == 0

    @pytest.mark.asyncio
    async def test_bulk_import_updates_existing_with_overwrite(
        self, async_client: AsyncClient, sample_dtc_codes, admin_auth_headers: dict
    ):
        """Test that bulk import updates existing codes when overwrite=true."""
        response = await async_client.post(
            "/api/v1/dtc/bulk",
            headers=admin_auth_headers,
            json={
                "codes": [
                    {
                        "code": "P0101",  # Already exists
                        "description_en": "Updated description",
                        "category": "powertrain",
                        "severity": "high",
                    },
                ],
                "overwrite_existing": True,
            },
        )

        assert response.status_code in (200, 201)
        data = response.json()

        assert data["updated"] == 1
        assert data["skipped"] == 0


class TestDTCResponseFormat:
    """Tests for DTC response format consistency."""

    @pytest.mark.asyncio
    async def test_search_result_format(self, async_client: AsyncClient, sample_dtc_codes):
        """Test search result has consistent format."""
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "P0101"},
        )

        assert response.status_code == 200
        data = response.json()

        if data:
            result = data[0]
            assert "code" in result
            assert "description_en" in result
            assert "category" in result
            assert "severity" in result
            assert "is_generic" in result
            assert "relevance_score" in result

    @pytest.mark.asyncio
    async def test_detail_result_format(self, async_client: AsyncClient, sample_dtc_codes):
        """Test detail result has consistent format."""
        response = await async_client.get("/api/v1/dtc/P0101")

        assert response.status_code == 200
        data = response.json()

        # Required fields
        assert "code" in data
        assert "description_en" in data
        assert "category" in data
        assert "severity" in data
        assert "is_generic" in data

        # Optional/additional fields
        assert "description_hu" in data
        assert "system" in data
        assert "symptoms" in data
        assert "possible_causes" in data
        assert "diagnostic_steps" in data
        assert "related_codes" in data

    @pytest.mark.asyncio
    async def test_symptoms_is_list(self, async_client: AsyncClient, sample_dtc_codes):
        """Test that symptoms field is a list."""
        response = await async_client.get("/api/v1/dtc/P0101")

        assert response.status_code == 200
        data = response.json()

        assert isinstance(data["symptoms"], list)

    @pytest.mark.asyncio
    async def test_possible_causes_is_list(self, async_client: AsyncClient, sample_dtc_codes):
        """Test that possible_causes field is a list."""
        response = await async_client.get("/api/v1/dtc/P0101")

        assert response.status_code == 200
        data = response.json()

        assert isinstance(data["possible_causes"], list)

    @pytest.mark.asyncio
    async def test_diagnostic_steps_is_list(self, async_client: AsyncClient, sample_dtc_codes):
        """Test that diagnostic_steps field is a list."""
        response = await async_client.get("/api/v1/dtc/P0101")

        assert response.status_code == 200
        data = response.json()

        assert isinstance(data["diagnostic_steps"], list)


# =============================================================================
# Search / detail / related must agree on what this API can serve
# =============================================================================
# The detail path validates through normalize_dtc_code (SAE J2012), so junk
# answers 400. The LEXICAL search path did not, and the shipped seed corpus
# still contains five rows that fail the rule - PEACE, PACED, P93AF, UA80E,
# UA80F, in data/dtc_codes/all_codes_{merged,complete}.json and
# backend/data/dtc_codes_seed.json. Result before this fix:
# GET /dtc/search?q=PEACE returned 200 with the row, and clicking it hit
# GET /dtc/PEACE -> 400. A listed, searchable, permanently un-openable result.
# =============================================================================

# The exact codes found in the shipped seed files.
SEEDED_JUNK_CODES = ["PEACE", "PACED", "P93AF", "UA80E", "UA80F"]


@pytest_asyncio.fixture
async def seeded_junk_dtc_codes(db_session: AsyncSession, sample_dtc_codes):
    """Insert the junk rows the seed corpus really contains.

    Written straight to the session rather than through the API, because the
    only way they got into production was a bulk import that predates the
    current validator - reproducing the state, not the route.
    """
    rows = []
    for i, code in enumerate(SEEDED_JUNK_CODES, start=900):
        row = DTCCode(
            id=i,
            code=code,
            description_en=f"Junk row {code} imported by the old loose pattern",
            description_hu=f"Ervenytelen sor {code}",
            category="powertrain",
            severity="medium",
            is_generic=True,
            system="Fuel and Air Metering",
            symptoms=[],
            possible_causes=[],
            diagnostic_steps=[],
            related_codes=["P0101"],
        )
        db_session.add(row)
        rows.append(row)
    await db_session.commit()
    return rows


class TestDTCApiSelfConsistency:
    @pytest.mark.parametrize("junk", SEEDED_JUNK_CODES)
    @pytest.mark.asyncio
    async def test_search_does_not_list_a_code_the_detail_endpoint_rejects(
        self, async_client: AsyncClient, seeded_junk_dtc_codes, junk: str
    ):
        """The regression: q=PEACE used to return the PEACE row with a 200."""
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": junk, "use_semantic": "false", "skip_cache": "true"},
        )

        assert response.status_code == 200
        assert junk not in [d["code"] for d in response.json()]

    @pytest.mark.asyncio
    async def test_a_description_search_does_not_surface_junk_either(
        self, async_client: AsyncClient, seeded_junk_dtc_codes
    ):
        """The ILIKE arm matches descriptions too, not just the code column."""
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "Junk row", "use_semantic": "false", "skip_cache": "true"},
        )

        assert response.status_code == 200
        assert [d["code"] for d in response.json()] == []

    @pytest.mark.asyncio
    async def test_the_semantic_arm_drops_junk_too(
        self, async_client: AsyncClient, seeded_junk_dtc_codes
    ):
        """A Qdrant payload code is just as untrusted as an ILIKE match."""
        hits = [
            {
                "id": 1,
                "score": 0.95,
                "payload": {"type": "dtc", "code": "UA80E", "category": "powertrain"},
            },
            {
                "id": 2,
                "score": 0.80,
                "payload": {"type": "dtc", "code": "P0101", "category": "powertrain"},
            },
        ]
        with (
            patch(
                "app.api.v1.endpoints.dtc_codes.qdrant_client",
                _qdrant_with_mocked_search(hits),
            ),
            patch(
                "app.api.v1.endpoints.dtc_codes.get_embedding_service",
                return_value=_embedding_service_stub(),
            ),
        ):
            response = await async_client.get(
                "/api/v1/dtc/search",
                params={"q": "motor rángatás", "skip_cache": "true"},
            )

        assert response.status_code == 200
        codes = [d["code"] for d in response.json()]
        assert "UA80E" not in codes
        assert "P0101" in codes  # the valid neighbour is untouched

    @pytest.mark.asyncio
    async def test_every_search_result_can_actually_be_opened(
        self, async_client: AsyncClient, seeded_junk_dtc_codes
    ):
        """The invariant, stated end-to-end: no result is a dead link."""
        search = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "P", "use_semantic": "false", "limit": 100, "skip_cache": "true"},
        )
        assert search.status_code == 200
        listed = [d["code"] for d in search.json()]
        assert listed, "need at least one result for this to prove anything"

        for code in listed:
            detail = await async_client.get(f"/api/v1/dtc/{code}", params={"skip_cache": "true"})
            assert detail.status_code != 400, f"search listed {code}, detail refuses it"

    @pytest.mark.parametrize("junk", SEEDED_JUNK_CODES)
    @pytest.mark.asyncio
    async def test_related_rejects_the_same_codes_as_detail(
        self, async_client: AsyncClient, seeded_junk_dtc_codes, junk: str
    ):
        """/related had NO format validation - it 404'd or served junk instead.

        Both siblings bind the same {code} path parameter; disagreeing on which
        spellings are legal let junk reach the Neo4j lookup by the back door.
        """
        detail = await async_client.get(f"/api/v1/dtc/{junk}", params={"skip_cache": "true"})
        related = await async_client.get(f"/api/v1/dtc/{junk}/related")

        assert detail.status_code == 400
        assert related.status_code == 400

    @pytest.mark.asyncio
    async def test_related_still_serves_valid_codes(
        self, async_client: AsyncClient, seeded_junk_dtc_codes
    ):
        """The new gate must not break the happy path, and must not suggest junk.

        P0101's stored related_codes and the P-prefix fallback both reach into
        the same table the junk rows live in.
        """
        response = await async_client.get("/api/v1/dtc/P0101/related", params={"limit": 50})

        assert response.status_code == 200
        codes = [d["code"] for d in response.json()]
        assert not set(codes) & set(SEEDED_JUNK_CODES)

    @pytest.mark.parametrize("code", ["P0101", "P26B7", "p0a94"])
    @pytest.mark.asyncio
    async def test_related_accepts_every_spelling_detail_accepts(
        self, async_client: AsyncClient, seeded_junk_dtc_codes, code: str
    ):
        """Agreement runs both ways: a real hex code must not be rejected."""
        related = await async_client.get(f"/api/v1/dtc/{code}/related")
        assert related.status_code != 400


class TestDTCWriteReadSymmetry:
    """The WRITE path must not accept what the READ path refuses.

    The read side was fixed first, which left the asymmetry that produced the
    seeded junk in the first place: `DTCCreate.code` only checked
    `5 <= len(code) <= 10`, so `POST /api/v1/dtc/` happily inserted `PEACE`,
    and `GET /api/v1/dtc/PEACE` then answered 400. A row you can create and
    cannot open.
    """

    @pytest.mark.parametrize("junk", SEEDED_JUNK_CODES)
    @pytest.mark.asyncio
    async def test_create_refuses_every_code_the_detail_endpoint_refuses(
        self, async_client: AsyncClient, admin_auth_headers: dict, junk: str
    ):
        response = await async_client.post(
            "/api/v1/dtc/",
            json={
                "code": junk,
                "description_en": "Junk row the old length-only check let through",
                "category": "powertrain",
                "severity": "medium",
            },
            headers=admin_auth_headers,
        )
        assert response.status_code == 422, f"{junk} was accepted by the write path"

        # ...and nothing landed: the code is still unknown, not merely invalid.
        detail = await async_client.get(f"/api/v1/dtc/{junk}", params={"skip_cache": "true"})
        assert detail.status_code == 400

    @pytest.mark.parametrize("code", ["P26B7", "p0a94", "B00A0", "U0100", " P0300 "])
    @pytest.mark.asyncio
    async def test_create_still_accepts_real_codes_and_they_open(
        self, async_client: AsyncClient, admin_auth_headers: dict, code: str
    ):
        """Not over-strict: hex codes, lower case and padding all round-trip.

        A rule that rejected P26B7 would be the previous bug in reverse - the
        strict `[PBCU][0-9]{4}` pattern this project already removed once.
        """
        response = await async_client.post(
            "/api/v1/dtc/",
            json={
                "code": code,
                "description_en": "Real code with a hex tail",
                "category": "powertrain",
                "severity": "medium",
            },
            headers=admin_auth_headers,
        )
        assert response.status_code == 201, response.text

        canonical = code.strip().upper()
        assert response.json()["code"] == canonical
        assert response.headers["Location"] == f"/api/v1/dtc/{canonical}"

        detail = await async_client.get(f"/api/v1/dtc/{canonical}", params={"skip_cache": "true"})
        assert detail.status_code == 200
        assert detail.json()["code"] == canonical

    @pytest.mark.asyncio
    async def test_bulk_import_cannot_seed_junk_either(
        self, async_client: AsyncClient, admin_auth_headers: dict
    ):
        """The bulk route shares DTCCreate, so it inherits the same gate.

        `POST /dtc/bulk` is the plausible way a corpus dump gets re-imported;
        leaving it lenient would have closed the door and left the window open.
        """
        response = await async_client.post(
            "/api/v1/dtc/bulk",
            json={
                "codes": [
                    {
                        "code": "P0101",
                        "description_en": "A perfectly good code",
                        "category": "powertrain",
                        "severity": "medium",
                    },
                    {
                        "code": "PEACE",
                        "description_en": "A hex-shaped English word",
                        "category": "powertrain",
                        "severity": "medium",
                    },
                ],
                "overwrite_existing": True,
            },
            headers=admin_auth_headers,
        )
        assert response.status_code == 422
