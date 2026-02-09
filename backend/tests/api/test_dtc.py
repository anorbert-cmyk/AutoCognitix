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

import pytest
from httpx import AsyncClient
from unittest.mock import patch, AsyncMock


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
        response = await async_client.get("/api/v1/dtc/P9999")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_dtc_detail_invalid_format_returns_400(self, async_client: AsyncClient):
        """Test getting DTC with invalid format returns 400."""
        response = await async_client.get("/api/v1/dtc/INVALID")

        assert response.status_code == 400

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
        response = await async_client.get("/api/v1/dtc/P9999/related")

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
    async def test_create_dtc_returns_201(self, async_client: AsyncClient, dtc_create_data: dict):
        """Test creating DTC returns 201."""
        response = await async_client.post(
            "/api/v1/dtc/",
            json=dtc_create_data,
        )

        assert response.status_code == 201
        data = response.json()
        assert data["code"] == dtc_create_data["code"].upper()

    @pytest.mark.asyncio
    async def test_create_dtc_normalizes_code_to_uppercase(self, async_client: AsyncClient):
        """Test that created DTC code is normalized to uppercase."""
        response = await async_client.post(
            "/api/v1/dtc/",
            json={
                "code": "p8888",  # lowercase
                "description_en": "Test code",
                "category": "powertrain",
                "severity": "medium",
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["code"] == "P8888"

    @pytest.mark.asyncio
    async def test_create_duplicate_dtc_returns_400(
        self, async_client: AsyncClient, sample_dtc_codes
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
        )

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_create_dtc_missing_required_fields_returns_422(self, async_client: AsyncClient):
        """Test creating DTC without required fields returns 422."""
        response = await async_client.post(
            "/api/v1/dtc/",
            json={"code": "P9999"},  # Missing description_en, category, severity
        )

        assert response.status_code == 422


class TestDTCBulkImport:
    """Tests for POST /api/v1/dtc/bulk endpoint."""

    @pytest.mark.asyncio
    async def test_bulk_import_returns_201(self, async_client: AsyncClient):
        """Test bulk import returns 201."""
        response = await async_client.post(
            "/api/v1/dtc/bulk",
            json={
                "codes": [
                    {
                        "code": "P7777",
                        "description_en": "Bulk test 1",
                        "category": "powertrain",
                        "severity": "low",
                    },
                    {
                        "code": "P7778",
                        "description_en": "Bulk test 2",
                        "category": "powertrain",
                        "severity": "medium",
                    },
                ],
                "overwrite_existing": False,
            },
        )

        assert response.status_code == 201
        data = response.json()

        assert "created" in data
        assert "updated" in data
        assert "skipped" in data
        assert "total" in data

    @pytest.mark.asyncio
    async def test_bulk_import_counts_created(self, async_client: AsyncClient):
        """Test that bulk import correctly counts created codes."""
        response = await async_client.post(
            "/api/v1/dtc/bulk",
            json={
                "codes": [
                    {
                        "code": "P6666",
                        "description_en": "New code 1",
                        "category": "powertrain",
                        "severity": "low",
                    },
                    {
                        "code": "P6667",
                        "description_en": "New code 2",
                        "category": "powertrain",
                        "severity": "low",
                    },
                ],
                "overwrite_existing": False,
            },
        )

        assert response.status_code == 201
        data = response.json()

        assert data["created"] == 2
        assert data["total"] == 2

    @pytest.mark.asyncio
    async def test_bulk_import_skips_existing_without_overwrite(
        self, async_client: AsyncClient, sample_dtc_codes
    ):
        """Test that bulk import skips existing codes when overwrite=false."""
        response = await async_client.post(
            "/api/v1/dtc/bulk",
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

        assert response.status_code == 201
        data = response.json()

        assert data["skipped"] == 1
        assert data["created"] == 0

    @pytest.mark.asyncio
    async def test_bulk_import_updates_existing_with_overwrite(
        self, async_client: AsyncClient, sample_dtc_codes
    ):
        """Test that bulk import updates existing codes when overwrite=true."""
        response = await async_client.post(
            "/api/v1/dtc/bulk",
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

        assert response.status_code == 201
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
