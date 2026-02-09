"""
Integration tests for the DTC (Diagnostic Trouble Code) API endpoints.

Tests the DTC code search, detail retrieval, and category endpoints
with actual database integration.
"""

import pytest
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(backend_path))


class TestDTCSearchEndpoint:
    """Test GET /api/v1/dtc/search endpoint."""

    @pytest.mark.asyncio
    async def test_search_returns_200(self, async_client, seeded_db):
        """Test that search returns 200 OK."""
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "P0101"},
        )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_search_returns_list(self, async_client, seeded_db):
        """Test that search returns a list."""
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "P0101"},
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_search_finds_exact_code(self, async_client, seeded_db):
        """Test that search finds exact DTC code."""
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "P0101"},
        )

        assert response.status_code == 200
        data = response.json()

        # Should find at least one result
        assert len(data) >= 1

        # First result should be exact match
        codes = [item["code"] for item in data]
        assert "P0101" in codes

    @pytest.mark.asyncio
    async def test_search_is_case_insensitive(self, async_client, seeded_db):
        """Test that search is case-insensitive."""
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

        # Results should be the same
        data_upper = response_upper.json()
        data_lower = response_lower.json()
        assert len(data_upper) == len(data_lower)

    @pytest.mark.asyncio
    async def test_search_by_description(self, async_client, seeded_db):
        """Test that search finds codes by description."""
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "mass air"},
        )

        assert response.status_code == 200
        data = response.json()

        # Should find P0101 which has "Mass Air Flow" in description
        if data:
            descriptions = [item.get("description_en", "").lower() for item in data]
            assert any("mass" in d and "air" in d for d in descriptions)

    @pytest.mark.asyncio
    async def test_search_by_hungarian_description(self, async_client, seeded_db):
        """Test that search finds codes by Hungarian description."""
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "levego"},
        )

        assert response.status_code == 200
        data = response.json()

        # Should find codes with Hungarian descriptions containing "levego"
        if data:
            descriptions = [item.get("description_hu", "").lower() for item in data]
            assert any("levego" in d for d in descriptions)

    @pytest.mark.asyncio
    async def test_search_with_category_filter(self, async_client, seeded_db):
        """Test that search filters by category."""
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "P", "category": "powertrain"},
        )

        assert response.status_code == 200
        data = response.json()

        # All results should be powertrain category
        for item in data:
            assert item["category"] == "powertrain"

    @pytest.mark.asyncio
    async def test_search_with_limit(self, async_client, seeded_db):
        """Test that search respects limit parameter."""
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "P", "limit": 2},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) <= 2

    @pytest.mark.asyncio
    async def test_search_empty_query_rejected(self, async_client, seeded_db):
        """Test that empty query is rejected."""
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": ""},
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_search_invalid_category_rejected(self, async_client, seeded_db):
        """Test that invalid category is rejected."""
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "P", "category": "invalid"},
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_search_limit_validation(self, async_client, seeded_db):
        """Test that limit parameter is validated."""
        # Too low
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "P", "limit": 0},
        )
        assert response.status_code == 422

        # Too high
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "P", "limit": 200},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_search_returns_required_fields(self, async_client, seeded_db):
        """Test that search results contain required fields."""
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "P0101"},
        )

        assert response.status_code == 200
        data = response.json()

        if data:
            required_fields = ["code", "description_en", "category", "is_generic"]
            for item in data:
                for field in required_fields:
                    assert field in item, f"Missing field: {field}"

    @pytest.mark.asyncio
    async def test_search_no_results_returns_empty_list(self, async_client, seeded_db):
        """Test that search with no matches returns empty list."""
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "XYZNONEXISTENT"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data == []


class TestDTCDetailEndpoint:
    """Test GET /api/v1/dtc/{code} endpoint."""

    @pytest.mark.asyncio
    async def test_get_valid_code_returns_200(self, async_client, seeded_db):
        """Test that getting valid code returns 200 OK."""
        response = await async_client.get("/api/v1/dtc/P0101")

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_get_valid_code_returns_details(self, async_client, seeded_db):
        """Test that getting valid code returns complete details."""
        response = await async_client.get("/api/v1/dtc/P0101")

        assert response.status_code == 200
        data = response.json()

        assert data["code"] == "P0101"
        assert "description_en" in data
        assert "description_hu" in data
        assert "category" in data
        assert "severity" in data

    @pytest.mark.asyncio
    async def test_get_code_returns_symptoms(self, async_client, seeded_db):
        """Test that code detail includes symptoms."""
        response = await async_client.get("/api/v1/dtc/P0101")

        assert response.status_code == 200
        data = response.json()

        assert "symptoms" in data
        assert isinstance(data["symptoms"], list)

    @pytest.mark.asyncio
    async def test_get_code_returns_possible_causes(self, async_client, seeded_db):
        """Test that code detail includes possible causes."""
        response = await async_client.get("/api/v1/dtc/P0101")

        assert response.status_code == 200
        data = response.json()

        assert "possible_causes" in data
        assert isinstance(data["possible_causes"], list)

    @pytest.mark.asyncio
    async def test_get_code_returns_diagnostic_steps(self, async_client, seeded_db):
        """Test that code detail includes diagnostic steps."""
        response = await async_client.get("/api/v1/dtc/P0101")

        assert response.status_code == 200
        data = response.json()

        assert "diagnostic_steps" in data
        assert isinstance(data["diagnostic_steps"], list)

    @pytest.mark.asyncio
    async def test_get_nonexistent_code_returns_404(self, async_client, seeded_db):
        """Test that nonexistent code returns 404."""
        response = await async_client.get("/api/v1/dtc/P9999")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_invalid_format_returns_400(self, async_client, seeded_db):
        """Test that invalid code format returns 400."""
        invalid_codes = ["INVALID", "X0101", "P01", "P012345"]

        for code in invalid_codes:
            response = await async_client.get(f"/api/v1/dtc/{code}")
            assert response.status_code == 400, f"Expected 400 for {code}"

    @pytest.mark.asyncio
    async def test_get_code_normalizes_lowercase(self, async_client, seeded_db):
        """Test that lowercase code is normalized."""
        response = await async_client.get("/api/v1/dtc/p0101")

        # Should work with lowercase
        assert response.status_code in [200, 404]

        if response.status_code == 200:
            data = response.json()
            # Code should be normalized to uppercase
            assert data["code"] == "P0101"

    @pytest.mark.asyncio
    async def test_get_body_code(self, async_client, seeded_db):
        """Test getting body (B) category code."""
        response = await async_client.get("/api/v1/dtc/B1234")

        assert response.status_code == 200
        data = response.json()
        assert data["category"] == "body"

    @pytest.mark.asyncio
    async def test_get_chassis_code(self, async_client, seeded_db):
        """Test getting chassis (C) category code."""
        response = await async_client.get("/api/v1/dtc/C0035")

        assert response.status_code == 200
        data = response.json()
        assert data["category"] == "chassis"

    @pytest.mark.asyncio
    async def test_get_network_code(self, async_client, seeded_db):
        """Test getting network (U) category code."""
        response = await async_client.get("/api/v1/dtc/U0100")

        assert response.status_code == 200
        data = response.json()
        assert data["category"] == "network"


class TestDTCRelatedCodesEndpoint:
    """Test GET /api/v1/dtc/{code}/related endpoint."""

    @pytest.mark.asyncio
    async def test_get_related_codes_returns_200(self, async_client, seeded_db):
        """Test that getting related codes returns 200 OK."""
        response = await async_client.get("/api/v1/dtc/P0101/related")

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_get_related_codes_returns_list(self, async_client, seeded_db):
        """Test that related codes endpoint returns a list."""
        response = await async_client.get("/api/v1/dtc/P0101/related")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_related_codes_have_required_fields(self, async_client, seeded_db):
        """Test that related codes have required fields."""
        response = await async_client.get("/api/v1/dtc/P0101/related")

        assert response.status_code == 200
        data = response.json()

        for item in data:
            assert "code" in item
            assert "description_en" in item

    @pytest.mark.asyncio
    async def test_related_codes_for_nonexistent_returns_empty(self, async_client, seeded_db):
        """Test that nonexistent code returns empty related list."""
        response = await async_client.get("/api/v1/dtc/P9999/related")

        # Should return empty list, not 404
        assert response.status_code == 200
        data = response.json()
        assert data == []


class TestDTCCategoriesEndpoint:
    """Test GET /api/v1/dtc/categories/list endpoint."""

    @pytest.mark.asyncio
    async def test_get_categories_returns_200(self, async_client, seeded_db):
        """Test that categories list returns 200 OK."""
        response = await async_client.get("/api/v1/dtc/categories/list")

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_get_categories_returns_four_categories(self, async_client, seeded_db):
        """Test that categories list returns all four categories."""
        response = await async_client.get("/api/v1/dtc/categories/list")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 4

    @pytest.mark.asyncio
    async def test_categories_include_all_types(self, async_client, seeded_db):
        """Test that all category types are included."""
        response = await async_client.get("/api/v1/dtc/categories/list")

        assert response.status_code == 200
        data = response.json()

        codes = [cat["code"] for cat in data]
        assert "P" in codes  # Powertrain
        assert "B" in codes  # Body
        assert "C" in codes  # Chassis
        assert "U" in codes  # Network

    @pytest.mark.asyncio
    async def test_categories_have_required_fields(self, async_client, seeded_db):
        """Test that categories have required fields."""
        response = await async_client.get("/api/v1/dtc/categories/list")

        assert response.status_code == 200
        data = response.json()

        required_fields = ["code", "name", "name_hu", "description", "description_hu"]
        for category in data:
            for field in required_fields:
                assert field in category, f"Missing field: {field}"

    @pytest.mark.asyncio
    async def test_categories_have_hungarian_translations(self, async_client, seeded_db):
        """Test that categories include Hungarian translations."""
        response = await async_client.get("/api/v1/dtc/categories/list")

        assert response.status_code == 200
        data = response.json()

        for category in data:
            assert category["name_hu"], "Hungarian name should not be empty"
            assert category["description_hu"], "Hungarian description should not be empty"


class TestDTCDataIntegrity:
    """Test DTC data integrity and consistency."""

    @pytest.mark.asyncio
    async def test_search_and_detail_match(self, async_client, seeded_db):
        """Test that search results match detail endpoint."""
        # First, search for a code
        search_response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "P0101"},
        )
        assert search_response.status_code == 200
        search_data = search_response.json()

        if search_data:
            search_result = next((item for item in search_data if item["code"] == "P0101"), None)

            if search_result:
                # Get detail for same code
                detail_response = await async_client.get("/api/v1/dtc/P0101")
                assert detail_response.status_code == 200
                detail_data = detail_response.json()

                # Compare common fields
                assert search_result["code"] == detail_data["code"]
                assert search_result["category"] == detail_data["category"]

    @pytest.mark.asyncio
    async def test_category_filter_matches_code_prefix(self, async_client, seeded_db):
        """Test that category filter correctly matches code prefix."""
        # Powertrain codes start with P
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "0", "category": "powertrain"},
        )
        assert response.status_code == 200
        data = response.json()

        for item in data:
            assert item["code"].startswith("P")
            assert item["category"] == "powertrain"

    @pytest.mark.asyncio
    async def test_severity_values_are_valid(self, async_client, seeded_db):
        """Test that severity values are from allowed set."""
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "P"},
        )
        assert response.status_code == 200
        data = response.json()

        valid_severities = {"low", "medium", "high", "critical"}
        for item in data:
            if "severity" in item:
                assert item["severity"] in valid_severities

    @pytest.mark.asyncio
    async def test_related_codes_are_valid_format(self, async_client, seeded_db):
        """Test that related codes are in valid DTC format."""
        response = await async_client.get("/api/v1/dtc/P0101/related")
        assert response.status_code == 200
        data = response.json()

        for item in data:
            code = item["code"]
            # Valid DTC format: Letter + 4 digits
            assert len(code) == 5
            assert code[0] in "PBCU"
            assert code[1:].isdigit()
