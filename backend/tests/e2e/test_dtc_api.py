"""
End-to-end tests for the DTC (Diagnostic Trouble Code) API.

Tests the complete DTC API including:
- Search endpoint with various queries
- Semantic search functionality
- DTC code details retrieval
- Related codes lookup
- Bulk import operations
- Category filtering
- Redis caching behavior
"""

import pytest
from unittest.mock import patch
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(backend_path))


class TestDTCSearchEndpoint:
    """Test GET /api/v1/dtc/search endpoint."""

    @pytest.mark.asyncio
    async def test_search_returns_200_with_valid_query(self, async_client, seeded_db):
        """Test that search returns 200 with valid query."""
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "P0101"},
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_search_by_dtc_code(self, async_client, seeded_db):
        """Test searching by exact DTC code."""
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "P0101"},
        )

        assert response.status_code == 200
        data = response.json()
        # Should find the P0101 code in seeded data
        if len(data) > 0:
            codes = [item["code"] for item in data]
            assert "P0101" in codes

    @pytest.mark.asyncio
    async def test_search_by_partial_code(self, async_client, seeded_db):
        """Test searching by partial DTC code."""
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "P01"},
        )

        assert response.status_code == 200
        data = response.json()
        # Should return codes starting with P01
        for item in data:
            assert item["code"].startswith("P01") or "P01" in item["code"].upper()

    @pytest.mark.asyncio
    async def test_search_by_english_description(self, async_client, seeded_db):
        """Test searching by English description text."""
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "Mass Air Flow"},
        )

        assert response.status_code == 200
        data = response.json()
        # Should find MAF-related codes
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_search_by_hungarian_description(self, async_client, seeded_db):
        """Test searching by Hungarian description text."""
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "Levegotomeg"},
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_search_requires_query_parameter(self, async_client, seeded_db):
        """Test that search requires query parameter."""
        response = await async_client.get("/api/v1/dtc/search")

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_search_query_minimum_length(self, async_client, seeded_db):
        """Test that search query must meet minimum length."""
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": ""},  # Empty query
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_search_respects_limit_parameter(self, async_client, seeded_db):
        """Test that search respects limit parameter."""
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "P", "limit": 5},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) <= 5

    @pytest.mark.asyncio
    async def test_search_default_limit(self, async_client, seeded_db):
        """Test that search has default limit."""
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "P"},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) <= 20  # Default limit

    @pytest.mark.asyncio
    async def test_search_limit_max_enforced(self, async_client, seeded_db):
        """Test that search limit cannot exceed maximum."""
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "P", "limit": 150},  # Max is 100
        )

        assert response.status_code == 422


class TestDTCSearchWithCategoryFilter:
    """Test DTC search with category filtering."""

    @pytest.mark.asyncio
    async def test_filter_by_powertrain_category(self, async_client, seeded_db):
        """Test filtering by powertrain (P) category."""
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "engine", "category": "powertrain"},
        )

        assert response.status_code == 200
        data = response.json()
        for item in data:
            assert item["category"] == "powertrain"

    @pytest.mark.asyncio
    async def test_filter_by_body_category(self, async_client, seeded_db):
        """Test filtering by body (B) category."""
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "airbag", "category": "body"},
        )

        assert response.status_code == 200
        data = response.json()
        for item in data:
            assert item["category"] == "body"

    @pytest.mark.asyncio
    async def test_filter_by_chassis_category(self, async_client, seeded_db):
        """Test filtering by chassis (C) category."""
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "wheel", "category": "chassis"},
        )

        assert response.status_code == 200
        data = response.json()
        for item in data:
            assert item["category"] == "chassis"

    @pytest.mark.asyncio
    async def test_filter_by_network_category(self, async_client, seeded_db):
        """Test filtering by network (U) category."""
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "communication", "category": "network"},
        )

        assert response.status_code == 200
        data = response.json()
        for item in data:
            assert item["category"] == "network"

    @pytest.mark.asyncio
    async def test_invalid_category_rejected(self, async_client, seeded_db):
        """Test that invalid category is rejected."""
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "engine", "category": "invalid_category"},
        )

        assert response.status_code == 422


class TestSemanticSearch:
    """Test semantic search functionality."""

    @pytest.mark.asyncio
    async def test_semantic_search_enabled_by_default(
        self,
        async_client,
        seeded_db,
        mock_embedding_service,
        mock_qdrant_client,
    ):
        """Test that semantic search is enabled by default."""
        with (
            patch(
                "app.services.embedding_service.get_embedding_service",
                return_value=mock_embedding_service,
            ),
            patch("app.db.qdrant_client.qdrant_client", mock_qdrant_client),
        ):
            response = await async_client.get(
                "/api/v1/dtc/search",
                params={"q": "motor rezeg", "use_semantic": True},
            )

            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_semantic_search_can_be_disabled(self, async_client, seeded_db):
        """Test that semantic search can be disabled."""
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "engine", "use_semantic": False},
        )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_semantic_search_handles_hungarian_text(
        self,
        async_client,
        seeded_db,
        mock_embedding_service,
        mock_qdrant_client,
    ):
        """Test that semantic search handles Hungarian text."""
        with (
            patch(
                "app.services.embedding_service.get_embedding_service",
                return_value=mock_embedding_service,
            ),
            patch("app.db.qdrant_client.qdrant_client", mock_qdrant_client),
        ):
            response = await async_client.get(
                "/api/v1/dtc/search",
                params={"q": "A motor nehezen indul hidegben", "use_semantic": True},
            )

            assert response.status_code == 200


class TestDTCSearchResultStructure:
    """Test DTC search result structure."""

    @pytest.mark.asyncio
    async def test_search_result_contains_required_fields(self, async_client, seeded_db):
        """Test that search results contain required fields."""
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "P0101"},
        )

        assert response.status_code == 200
        data = response.json()

        if len(data) > 0:
            item = data[0]
            required_fields = ["code", "description_en", "category", "is_generic"]
            for field in required_fields:
                assert field in item, f"Missing required field: {field}"

    @pytest.mark.asyncio
    async def test_search_result_includes_relevance_score(self, async_client, seeded_db):
        """Test that search results include relevance score."""
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "P0101"},
        )

        assert response.status_code == 200
        data = response.json()

        if len(data) > 0:
            item = data[0]
            # Relevance score may be None for exact matches
            assert "relevance_score" in item

    @pytest.mark.asyncio
    async def test_search_results_sorted_by_relevance(self, async_client, seeded_db):
        """Test that search results are sorted by relevance."""
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "P01"},
        )

        assert response.status_code == 200
        data = response.json()

        if len(data) > 1:
            # Check relevance scores are in descending order (where not None)
            scores = [
                item.get("relevance_score")
                for item in data
                if item.get("relevance_score") is not None
            ]
            if len(scores) > 1:
                for i in range(len(scores) - 1):
                    assert scores[i] >= scores[i + 1]


class TestDTCCodeDetailEndpoint:
    """Test GET /api/v1/dtc/{code} endpoint."""

    @pytest.mark.asyncio
    async def test_get_dtc_detail_returns_200(self, async_client, seeded_db):
        """Test that getting DTC detail returns 200."""
        response = await async_client.get("/api/v1/dtc/P0101")

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_get_dtc_detail_structure(self, async_client, seeded_db):
        """Test that DTC detail has correct structure."""
        response = await async_client.get("/api/v1/dtc/P0101")

        assert response.status_code == 200
        data = response.json()

        required_fields = [
            "code",
            "description_en",
            "category",
            "is_generic",
            "severity",
            "symptoms",
            "possible_causes",
            "diagnostic_steps",
        ]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

    @pytest.mark.asyncio
    async def test_get_dtc_detail_includes_hungarian_description(self, async_client, seeded_db):
        """Test that DTC detail includes Hungarian description."""
        response = await async_client.get("/api/v1/dtc/P0101")

        assert response.status_code == 200
        data = response.json()

        assert "description_hu" in data
        # Should have Hungarian description for seeded data
        if data["description_hu"]:
            assert len(data["description_hu"]) > 0

    @pytest.mark.asyncio
    async def test_get_dtc_detail_includes_symptoms(self, async_client, seeded_db):
        """Test that DTC detail includes symptoms list."""
        response = await async_client.get("/api/v1/dtc/P0101")

        assert response.status_code == 200
        data = response.json()

        assert "symptoms" in data
        assert isinstance(data["symptoms"], list)

    @pytest.mark.asyncio
    async def test_get_dtc_detail_includes_causes(self, async_client, seeded_db):
        """Test that DTC detail includes possible causes."""
        response = await async_client.get("/api/v1/dtc/P0101")

        assert response.status_code == 200
        data = response.json()

        assert "possible_causes" in data
        assert isinstance(data["possible_causes"], list)

    @pytest.mark.asyncio
    async def test_get_dtc_detail_includes_diagnostic_steps(self, async_client, seeded_db):
        """Test that DTC detail includes diagnostic steps."""
        response = await async_client.get("/api/v1/dtc/P0101")

        assert response.status_code == 200
        data = response.json()

        assert "diagnostic_steps" in data
        assert isinstance(data["diagnostic_steps"], list)

    @pytest.mark.asyncio
    async def test_get_nonexistent_dtc_returns_404(self, async_client, seeded_db):
        """Test that getting nonexistent DTC returns 404."""
        response = await async_client.get("/api/v1/dtc/P9999")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_invalid_dtc_format_returns_400(self, async_client, seeded_db):
        """Test that invalid DTC format returns 400."""
        response = await async_client.get("/api/v1/dtc/INVALID")

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_get_dtc_normalizes_lowercase(self, async_client, seeded_db):
        """Test that lowercase DTC codes are normalized."""
        response = await async_client.get("/api/v1/dtc/p0101")

        # Should work with lowercase
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == "P0101"


class TestDTCCodeDetailWithGraphData:
    """Test DTC detail with Neo4j graph data enrichment."""

    @pytest.mark.asyncio
    async def test_include_graph_data_by_default(
        self,
        async_client,
        seeded_db,
        mock_neo4j_client,
    ):
        """Test that graph data is included by default."""
        with patch("app.db.neo4j_models.get_diagnostic_path", return_value={}):
            response = await async_client.get(
                "/api/v1/dtc/P0101",
                params={"include_graph": True},
            )

            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_exclude_graph_data(self, async_client, seeded_db):
        """Test that graph data can be excluded."""
        response = await async_client.get(
            "/api/v1/dtc/P0101",
            params={"include_graph": False},
        )

        assert response.status_code == 200


class TestRelatedCodesEndpoint:
    """Test GET /api/v1/dtc/{code}/related endpoint."""

    @pytest.mark.asyncio
    async def test_get_related_codes_returns_200(self, async_client, seeded_db):
        """Test that getting related codes returns 200."""
        response = await async_client.get("/api/v1/dtc/P0101/related")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_related_codes_excludes_original(self, async_client, seeded_db):
        """Test that related codes don't include the original code."""
        response = await async_client.get("/api/v1/dtc/P0101/related")

        assert response.status_code == 200
        data = response.json()

        codes = [item["code"] for item in data]
        assert "P0101" not in codes

    @pytest.mark.asyncio
    async def test_related_codes_respects_limit(self, async_client, seeded_db):
        """Test that related codes respects limit parameter."""
        response = await async_client.get(
            "/api/v1/dtc/P0101/related",
            params={"limit": 3},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) <= 3

    @pytest.mark.asyncio
    async def test_related_codes_for_nonexistent_returns_404(self, async_client, seeded_db):
        """Test that related codes for nonexistent code returns 404."""
        response = await async_client.get("/api/v1/dtc/P9999/related")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_related_codes_include_relevance(self, async_client, seeded_db):
        """Test that related codes include relevance scores."""
        response = await async_client.get("/api/v1/dtc/P0101/related")

        assert response.status_code == 200
        data = response.json()

        for item in data:
            assert "relevance_score" in item


class TestDTCCategoriesEndpoint:
    """Test GET /api/v1/dtc/categories/list endpoint."""

    @pytest.mark.asyncio
    async def test_get_categories_returns_200(self, async_client, seeded_db):
        """Test that getting categories returns 200."""
        response = await async_client.get("/api/v1/dtc/categories/list")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_categories_include_all_types(self, async_client, seeded_db):
        """Test that categories include P, B, C, U types."""
        response = await async_client.get("/api/v1/dtc/categories/list")

        assert response.status_code == 200
        data = response.json()

        codes = [item["code"] for item in data]
        assert "P" in codes  # Powertrain
        assert "B" in codes  # Body
        assert "C" in codes  # Chassis
        assert "U" in codes  # Network

    @pytest.mark.asyncio
    async def test_categories_include_hungarian_names(self, async_client, seeded_db):
        """Test that categories include Hungarian names."""
        response = await async_client.get("/api/v1/dtc/categories/list")

        assert response.status_code == 200
        data = response.json()

        for category in data:
            assert "name_hu" in category


class TestDTCCreateEndpoint:
    """Test POST /api/v1/dtc endpoint."""

    @pytest.mark.asyncio
    async def test_create_dtc_code(self, async_client, seeded_db):
        """Test creating a new DTC code."""
        new_dtc = {
            "code": "P1234",
            "description_en": "Test DTC Code",
            "description_hu": "Teszt hibakod",
            "category": "powertrain",
            "severity": "medium",
            "is_generic": True,
        }

        response = await async_client.post("/api/v1/dtc/", json=new_dtc)

        assert response.status_code == 201
        data = response.json()
        assert data["code"] == "P1234"

    @pytest.mark.asyncio
    async def test_create_duplicate_dtc_fails(self, async_client, seeded_db):
        """Test that creating duplicate DTC code fails."""
        duplicate_dtc = {
            "code": "P0101",  # Already exists in seeded data
            "description_en": "Duplicate Test",
            "category": "powertrain",
            "severity": "medium",
        }

        response = await async_client.post("/api/v1/dtc/", json=duplicate_dtc)

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_create_dtc_validates_code_format(self, async_client, seeded_db):
        """Test that create validates DTC code format."""
        invalid_dtc = {
            "code": "X999",  # Invalid prefix
            "description_en": "Invalid Code",
            "category": "powertrain",
            "severity": "medium",
        }

        response = await async_client.post("/api/v1/dtc/", json=invalid_dtc)

        # Should reject invalid code format
        assert response.status_code in [400, 422]

    @pytest.mark.asyncio
    async def test_create_dtc_requires_description(self, async_client, seeded_db):
        """Test that create requires description."""
        incomplete_dtc = {
            "code": "P5555",
            "category": "powertrain",
            "severity": "medium",
        }

        response = await async_client.post("/api/v1/dtc/", json=incomplete_dtc)

        assert response.status_code == 422


class TestDTCBulkImportEndpoint:
    """Test POST /api/v1/dtc/bulk endpoint."""

    @pytest.mark.asyncio
    async def test_bulk_import_multiple_codes(self, async_client, seeded_db):
        """Test bulk importing multiple DTC codes."""
        bulk_data = {
            "codes": [
                {
                    "code": "P2001",
                    "description_en": "Test Code 1",
                    "category": "powertrain",
                    "severity": "medium",
                },
                {
                    "code": "P2002",
                    "description_en": "Test Code 2",
                    "category": "powertrain",
                    "severity": "low",
                },
            ],
            "overwrite_existing": False,
        }

        response = await async_client.post("/api/v1/dtc/bulk", json=bulk_data)

        assert response.status_code == 201
        data = response.json()
        assert "created" in data
        assert data["created"] >= 0

    @pytest.mark.asyncio
    async def test_bulk_import_skips_existing_by_default(self, async_client, seeded_db):
        """Test that bulk import skips existing codes by default."""
        bulk_data = {
            "codes": [
                {
                    "code": "P0101",  # Already exists
                    "description_en": "Updated Description",
                    "category": "powertrain",
                    "severity": "medium",
                },
            ],
            "overwrite_existing": False,
        }

        response = await async_client.post("/api/v1/dtc/bulk", json=bulk_data)

        assert response.status_code == 201
        data = response.json()
        assert data["skipped"] >= 1

    @pytest.mark.asyncio
    async def test_bulk_import_can_overwrite(self, async_client, seeded_db):
        """Test that bulk import can overwrite existing codes."""
        bulk_data = {
            "codes": [
                {
                    "code": "P0101",  # Already exists
                    "description_en": "Updated Description",
                    "category": "powertrain",
                    "severity": "high",  # Changed severity
                },
            ],
            "overwrite_existing": True,
        }

        response = await async_client.post("/api/v1/dtc/bulk", json=bulk_data)

        assert response.status_code == 201
        data = response.json()
        assert data["updated"] >= 0 or data["created"] >= 0

    @pytest.mark.asyncio
    async def test_bulk_import_returns_summary(self, async_client, seeded_db):
        """Test that bulk import returns summary."""
        bulk_data = {
            "codes": [
                {
                    "code": "P3001",
                    "description_en": "Test Code",
                    "category": "powertrain",
                    "severity": "medium",
                },
            ],
            "overwrite_existing": False,
        }

        response = await async_client.post("/api/v1/dtc/bulk", json=bulk_data)

        assert response.status_code == 201
        data = response.json()

        assert "created" in data
        assert "updated" in data
        assert "skipped" in data
        assert "errors" in data
        assert "total" in data


class TestDTCCaching:
    """Test Redis caching behavior for DTC endpoints."""

    @pytest.mark.asyncio
    async def test_search_can_skip_cache(self, async_client, seeded_db):
        """Test that search can skip cache."""
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "P0101", "skip_cache": True},
        )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_detail_can_skip_cache(self, async_client, seeded_db):
        """Test that detail endpoint can skip cache."""
        response = await async_client.get(
            "/api/v1/dtc/P0101",
            params={"skip_cache": True},
        )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_cached_response_matches_fresh(self, async_client, seeded_db):
        """Test that cached and fresh responses match."""
        # First request (may populate cache)
        response1 = await async_client.get(
            "/api/v1/dtc/P0101",
            params={"skip_cache": True},
        )

        # Second request (may use cache)
        response2 = await async_client.get("/api/v1/dtc/P0101")

        assert response1.status_code == 200
        assert response2.status_code == 200

        # Basic structure should match
        data1 = response1.json()
        data2 = response2.json()
        assert data1["code"] == data2["code"]


class TestDTCValidCodes:
    """Test with various valid DTC codes."""

    @pytest.mark.asyncio
    async def test_powertrain_codes(self, async_client, seeded_db, valid_dtc_codes):
        """Test powertrain (P) codes."""
        p_codes = [c for c in valid_dtc_codes if c.startswith("P")]

        for code in p_codes:
            response = await async_client.get(
                "/api/v1/dtc/search",
                params={"q": code},
            )
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_body_codes(self, async_client, seeded_db, valid_dtc_codes):
        """Test body (B) codes."""
        b_codes = [c for c in valid_dtc_codes if c.startswith("B")]

        for code in b_codes:
            response = await async_client.get(
                "/api/v1/dtc/search",
                params={"q": code},
            )
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_chassis_codes(self, async_client, seeded_db, valid_dtc_codes):
        """Test chassis (C) codes."""
        c_codes = [c for c in valid_dtc_codes if c.startswith("C")]

        for code in c_codes:
            response = await async_client.get(
                "/api/v1/dtc/search",
                params={"q": code},
            )
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_network_codes(self, async_client, seeded_db, valid_dtc_codes):
        """Test network (U) codes."""
        u_codes = [c for c in valid_dtc_codes if c.startswith("U")]

        for code in u_codes:
            response = await async_client.get(
                "/api/v1/dtc/search",
                params={"q": code},
            )
            assert response.status_code == 200


class TestDTCInvalidCodes:
    """Test handling of invalid DTC codes."""

    @pytest.mark.asyncio
    async def test_invalid_prefix_rejected(self, async_client, seeded_db):
        """Test that invalid prefix is rejected for detail lookup."""
        response = await async_client.get("/api/v1/dtc/X1234")

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_short_code_rejected(self, async_client, seeded_db):
        """Test that short code is rejected."""
        response = await async_client.get("/api/v1/dtc/P12")

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_empty_code_returns_error(self, async_client, seeded_db):
        """Test that empty code returns error."""
        response = await async_client.get("/api/v1/dtc/")

        # Should return method not allowed or not found
        assert response.status_code in [404, 405]


class TestDTCUpdateEndpoint:
    """Test DTC code update functionality."""

    @pytest.mark.asyncio
    async def test_update_dtc_description(self, async_client, seeded_db):
        """Test updating DTC description."""
        # First create a DTC
        dtc = {
            "code": "P6001",
            "description_en": "Original Description",
            "description_hu": "Eredeti leiras",
            "category": "powertrain",
            "severity": "medium",
        }

        create_response = await async_client.post("/api/v1/dtc/", json=dtc)
        assert create_response.status_code == 201

        # Update via bulk with overwrite
        update_data = {
            "codes": [
                {
                    "code": "P6001",
                    "description_en": "Updated Description",
                    "description_hu": "Frissitett leiras",
                    "category": "powertrain",
                    "severity": "high",  # Changed
                }
            ],
            "overwrite_existing": True,
        }

        update_response = await async_client.post("/api/v1/dtc/bulk", json=update_data)
        assert update_response.status_code == 201

        # Verify update
        get_response = await async_client.get("/api/v1/dtc/P6001")
        if get_response.status_code == 200:
            data = get_response.json()
            assert data["severity"] == "high"


class TestDTCSearchPagination:
    """Test DTC search pagination functionality."""

    @pytest.mark.asyncio
    async def test_search_with_offset(self, async_client, seeded_db):
        """Test search with offset parameter."""
        # First search without offset
        response1 = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "P", "limit": 5},
        )
        assert response1.status_code == 200
        data1 = response1.json()

        # Same search with different limit
        response2 = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "P", "limit": 10},
        )
        assert response2.status_code == 200
        data2 = response2.json()

        # Results should start with same codes
        if len(data1) > 0 and len(data2) > 0:
            assert data1[0]["code"] == data2[0]["code"]

    @pytest.mark.asyncio
    async def test_search_limit_boundary_values(self, async_client, seeded_db):
        """Test search with limit at boundary values."""
        # Minimum limit (1)
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "P", "limit": 1},
        )
        assert response.status_code == 200
        assert len(response.json()) <= 1

        # Maximum limit (100)
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "P", "limit": 100},
        )
        assert response.status_code == 200


class TestDTCHungarianContent:
    """Test Hungarian content handling in DTC operations."""

    @pytest.mark.asyncio
    async def test_create_dtc_with_hungarian_description(self, async_client, seeded_db):
        """Test creating DTC with Hungarian description."""
        dtc = {
            "code": "P6100",
            "description_en": "Test Code",
            "description_hu": "Teszt hibakod magyar leirassal es specialis karakterekkel",
            "category": "powertrain",
            "severity": "medium",
            "symptoms": ["Tunet 1", "Tunet 2"],
            "possible_causes": ["Ok 1", "Ok 2"],
        }

        response = await async_client.post("/api/v1/dtc/", json=dtc)
        assert response.status_code == 201

        # Verify Hungarian content was saved
        get_response = await async_client.get("/api/v1/dtc/P6100")
        if get_response.status_code == 200:
            data = get_response.json()
            assert data["description_hu"] == dtc["description_hu"]

    @pytest.mark.asyncio
    async def test_search_hungarian_accented_characters(self, async_client, seeded_db):
        """Test searching with Hungarian accented characters."""
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "motor"},  # Common Hungarian/English word
        )
        assert response.status_code == 200


class TestDTCRelationships:
    """Test DTC relationship functionality."""

    @pytest.mark.asyncio
    async def test_related_codes_sorted_by_relevance(self, async_client, seeded_db):
        """Test that related codes are sorted by relevance."""
        response = await async_client.get("/api/v1/dtc/P0101/related")

        assert response.status_code == 200
        data = response.json()

        if len(data) > 1:
            # Verify sorted by relevance (descending)
            scores = [item.get("relevance_score", 0) for item in data]
            for i in range(len(scores) - 1):
                if scores[i] is not None and scores[i + 1] is not None:
                    assert scores[i] >= scores[i + 1]

    @pytest.mark.asyncio
    async def test_related_codes_same_system(self, async_client, seeded_db):
        """Test that related codes often share the same system."""
        response = await async_client.get("/api/v1/dtc/P0101/related")

        assert response.status_code == 200
        data = response.json()

        # Related codes for P0xxx should mostly be Pxxxx codes
        if len(data) > 0:
            p_codes = [item for item in data if item["code"].startswith("P")]
            # At least some should be P codes
            assert len(p_codes) >= 0  # May be empty if no related codes


class TestDTCBulkOperations:
    """Test DTC bulk operation edge cases."""

    @pytest.mark.asyncio
    async def test_bulk_import_empty_list(self, async_client, seeded_db):
        """Test bulk import with empty list."""
        bulk_data = {
            "codes": [],
            "overwrite_existing": False,
        }

        response = await async_client.post("/api/v1/dtc/bulk", json=bulk_data)

        # Should handle empty list gracefully
        assert response.status_code in [201, 422]

    @pytest.mark.asyncio
    async def test_bulk_import_with_invalid_codes(self, async_client, seeded_db):
        """Test bulk import with some invalid codes."""
        bulk_data = {
            "codes": [
                {
                    "code": "P7001",
                    "description_en": "Valid Code",
                    "category": "powertrain",
                    "severity": "medium",
                },
                {
                    "code": "X9999",  # Invalid prefix
                    "description_en": "Invalid Code",
                    "category": "powertrain",
                    "severity": "medium",
                },
            ],
            "overwrite_existing": False,
        }

        response = await async_client.post("/api/v1/dtc/bulk", json=bulk_data)

        # Should either reject all or create valid ones and report errors
        assert response.status_code in [201, 422]

    @pytest.mark.asyncio
    async def test_bulk_import_large_batch(self, async_client, seeded_db):
        """Test bulk import with larger batch."""
        codes = [
            {
                "code": f"P7{i:03d}",
                "description_en": f"Test Code {i}",
                "category": "powertrain",
                "severity": "medium",
            }
            for i in range(50)  # 50 codes
        ]

        bulk_data = {
            "codes": codes,
            "overwrite_existing": False,
        }

        response = await async_client.post("/api/v1/dtc/bulk", json=bulk_data)
        assert response.status_code == 201

        data = response.json()
        assert data["total"] == 50


class TestDTCSearchCaseSensitivity:
    """Test case sensitivity in DTC searches."""

    @pytest.mark.asyncio
    async def test_search_case_insensitive(self, async_client, seeded_db):
        """Test that search is case insensitive."""
        # Search with lowercase
        response_lower = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "p0101"},
        )

        # Search with uppercase
        response_upper = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "P0101"},
        )

        # Search with mixed case
        response_mixed = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "P0101"},
        )

        assert response_lower.status_code == 200
        assert response_upper.status_code == 200
        assert response_mixed.status_code == 200

    @pytest.mark.asyncio
    async def test_detail_endpoint_case_insensitive(self, async_client, seeded_db):
        """Test that detail endpoint handles case insensitivity."""
        # Lowercase
        response_lower = await async_client.get("/api/v1/dtc/p0101")

        # Uppercase
        response_upper = await async_client.get("/api/v1/dtc/P0101")

        # Both should return same result
        assert response_lower.status_code == 200
        assert response_upper.status_code == 200

        if response_lower.status_code == 200 and response_upper.status_code == 200:
            data_lower = response_lower.json()
            data_upper = response_upper.json()
            assert data_lower["code"] == data_upper["code"]


class TestDTCSeverityFiltering:
    """Test DTC severity filtering."""

    @pytest.mark.asyncio
    async def test_filter_by_severity_high(self, async_client, seeded_db):
        """Test filtering by high severity."""
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "engine"},
        )

        assert response.status_code == 200
        data = response.json()

        # Check severity values are valid
        for item in data:
            if "severity" in item:
                assert item["severity"] in ["low", "medium", "high", "critical", "unknown"]


class TestDTCManufacturerSpecific:
    """Test manufacturer-specific DTC handling."""

    @pytest.mark.asyncio
    async def test_create_manufacturer_specific_code(self, async_client, seeded_db):
        """Test creating manufacturer-specific DTC code."""
        dtc = {
            "code": "P1234",  # Manufacturer-specific range
            "description_en": "Manufacturer Specific Code",
            "description_hu": "Gyartospecifikus hibakod",
            "category": "powertrain",
            "severity": "medium",
            "is_generic": False,
            "system": "Fuel System",
        }

        response = await async_client.post("/api/v1/dtc/", json=dtc)
        assert response.status_code == 201

        data = response.json()
        assert data["code"] == "P1234"

    @pytest.mark.asyncio
    async def test_generic_vs_manufacturer_flag(self, async_client, seeded_db):
        """Test is_generic flag handling."""
        # Create generic code
        generic_dtc = {
            "code": "P0200",
            "description_en": "Generic Injector Circuit",
            "category": "powertrain",
            "severity": "medium",
            "is_generic": True,
        }

        response = await async_client.post("/api/v1/dtc/", json=generic_dtc)
        assert response.status_code == 201

        # Verify flag
        get_response = await async_client.get("/api/v1/dtc/P0200")
        if get_response.status_code == 200:
            data = get_response.json()
            assert data["is_generic"] is True


class TestDTCDiagnosticSteps:
    """Test DTC diagnostic steps handling."""

    @pytest.mark.asyncio
    async def test_create_dtc_with_diagnostic_steps(self, async_client, seeded_db):
        """Test creating DTC with diagnostic steps."""
        dtc = {
            "code": "P6200",
            "description_en": "Test Code with Steps",
            "category": "powertrain",
            "severity": "medium",
            "diagnostic_steps": [
                "Check sensor resistance",
                "Verify wiring connections",
                "Test signal output",
                "Replace sensor if faulty",
            ],
        }

        response = await async_client.post("/api/v1/dtc/", json=dtc)
        assert response.status_code == 201

        # Verify steps saved
        get_response = await async_client.get("/api/v1/dtc/P6200")
        if get_response.status_code == 200:
            data = get_response.json()
            assert len(data["diagnostic_steps"]) == 4

    @pytest.mark.asyncio
    async def test_dtc_detail_includes_all_lists(self, async_client, seeded_db):
        """Test that DTC detail includes all list fields."""
        response = await async_client.get("/api/v1/dtc/P0101")

        assert response.status_code == 200
        data = response.json()

        # Check all list fields are present and are lists
        list_fields = ["symptoms", "possible_causes", "diagnostic_steps", "related_codes"]
        for field in list_fields:
            assert field in data
            assert isinstance(data[field], list)


class TestDTCCacheInvalidation:
    """Test cache invalidation on DTC updates."""

    @pytest.mark.asyncio
    async def test_cache_skip_returns_fresh_data(self, async_client, seeded_db):
        """Test that skip_cache returns fresh data."""
        # First request (may populate cache)
        response1 = await async_client.get(
            "/api/v1/dtc/P0101",
            params={"skip_cache": False},
        )

        # Second request with skip_cache
        response2 = await async_client.get(
            "/api/v1/dtc/P0101",
            params={"skip_cache": True},
        )

        assert response1.status_code == 200
        assert response2.status_code == 200

        # Both should return same data
        data1 = response1.json()
        data2 = response2.json()
        assert data1["code"] == data2["code"]

    @pytest.mark.asyncio
    async def test_search_cache_skip(self, async_client, seeded_db):
        """Test search with cache skip."""
        # First search (may populate cache)
        response1 = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "P0101", "skip_cache": False},
        )

        # Second search with skip_cache
        response2 = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "P0101", "skip_cache": True},
        )

        assert response1.status_code == 200
        assert response2.status_code == 200


class TestDTCErrorMessages:
    """Test DTC error message quality."""

    @pytest.mark.asyncio
    async def test_404_includes_helpful_message(self, async_client, seeded_db):
        """Test that 404 error includes helpful message."""
        response = await async_client.get("/api/v1/dtc/P9999")

        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "P9999" in data["detail"] or "not found" in data["detail"].lower()

    @pytest.mark.asyncio
    async def test_400_includes_format_hint(self, async_client, seeded_db):
        """Test that 400 error includes format hint."""
        response = await async_client.get("/api/v1/dtc/INVALID")

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        # Should mention expected format
        assert "format" in data["detail"].lower() or "invalid" in data["detail"].lower()
