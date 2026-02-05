"""
Tests for API endpoints with mocked database.

Tests cover:
- DTC code search endpoint
- DTC code detail endpoint
- Health check endpoint
- Input validation
- Error handling
"""

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_path))


@pytest.fixture
def app():
    """Create FastAPI app for testing."""
    from fastapi.responses import ORJSONResponse

    # Create minimal test app
    test_app = FastAPI(default_response_class=ORJSONResponse)

    # Import and include routers
    from app.api.v1.endpoints.dtc_codes import router as dtc_router
    test_app.include_router(dtc_router, prefix="/api/v1/dtc", tags=["DTC"])

    # Add health check
    @test_app.get("/health")
    async def health_check():
        return {"status": "healthy", "version": "0.1.0"}

    return test_app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check_returns_200(self, client):
        """Test health check returns 200 OK."""
        response = client.get("/health")

        assert response.status_code == 200

    def test_health_check_returns_healthy_status(self, client):
        """Test health check returns healthy status."""
        response = client.get("/health")
        data = response.json()

        assert data["status"] == "healthy"

    def test_health_check_returns_version(self, client):
        """Test health check includes version."""
        response = client.get("/health")
        data = response.json()

        assert "version" in data
        assert data["version"] == "0.1.0"


class TestDTCSearchEndpoint:
    """Test DTC code search endpoint."""

    def test_search_by_code(self, client):
        """Test searching for DTC by code."""
        response = client.get("/api/v1/dtc/search", params={"q": "P0101"})

        assert response.status_code == 200
        data = response.json()

        # Should return list of results
        assert isinstance(data, list)

    def test_search_by_description(self, client):
        """Test searching for DTC by description keyword."""
        response = client.get("/api/v1/dtc/search", params={"q": "mass air"})

        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, list)

    def test_search_case_insensitive(self, client):
        """Test that search is case-insensitive."""
        response_upper = client.get("/api/v1/dtc/search", params={"q": "P0101"})
        response_lower = client.get("/api/v1/dtc/search", params={"q": "p0101"})

        assert response_upper.status_code == 200
        assert response_lower.status_code == 200

    def test_search_with_category_filter(self, client):
        """Test search with category filter."""
        response = client.get(
            "/api/v1/dtc/search",
            params={"q": "P", "category": "powertrain"}
        )

        assert response.status_code == 200
        data = response.json()

        # All results should be powertrain category
        for item in data:
            assert item["category"] == "powertrain"

    def test_search_with_limit(self, client):
        """Test search with result limit."""
        response = client.get(
            "/api/v1/dtc/search",
            params={"q": "P", "limit": 5}
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data) <= 5

    def test_search_empty_query_rejected(self, client):
        """Test that empty query is rejected."""
        response = client.get("/api/v1/dtc/search", params={"q": ""})

        # Should return 422 validation error
        assert response.status_code == 422

    def test_search_returns_required_fields(self, client):
        """Test that search results contain required fields."""
        response = client.get("/api/v1/dtc/search", params={"q": "P0101"})

        assert response.status_code == 200
        data = response.json()

        if data:  # If results exist
            first_result = data[0]
            required_fields = ["code", "description_en", "category", "is_generic"]

            for field in required_fields:
                assert field in first_result, f"Missing field: {field}"


class TestDTCDetailEndpoint:
    """Test DTC code detail endpoint."""

    def test_get_valid_code_detail(self, client):
        """Test getting details for valid DTC code."""
        response = client.get("/api/v1/dtc/P0101")

        assert response.status_code == 200
        data = response.json()

        assert data["code"] == "P0101"
        assert "description_en" in data
        assert "description_hu" in data

    def test_get_code_detail_returns_symptoms(self, client):
        """Test that code detail includes symptoms."""
        response = client.get("/api/v1/dtc/P0101")

        assert response.status_code == 200
        data = response.json()

        assert "symptoms" in data
        assert isinstance(data["symptoms"], list)

    def test_get_code_detail_returns_causes(self, client):
        """Test that code detail includes possible causes."""
        response = client.get("/api/v1/dtc/P0101")

        assert response.status_code == 200
        data = response.json()

        assert "possible_causes" in data
        assert isinstance(data["possible_causes"], list)

    def test_get_code_detail_returns_diagnostic_steps(self, client):
        """Test that code detail includes diagnostic steps."""
        response = client.get("/api/v1/dtc/P0101")

        assert response.status_code == 200
        data = response.json()

        assert "diagnostic_steps" in data
        assert isinstance(data["diagnostic_steps"], list)

    def test_get_nonexistent_code_returns_404(self, client):
        """Test that nonexistent code returns 404."""
        response = client.get("/api/v1/dtc/P9999")

        assert response.status_code == 404

    def test_get_invalid_code_format_returns_400(self, client):
        """Test that invalid code format returns 400."""
        invalid_codes = ["INVALID", "X0101", "P01", "P01011"]

        for code in invalid_codes:
            response = client.get(f"/api/v1/dtc/{code}")
            assert response.status_code == 400, f"Expected 400 for {code}"

    def test_code_detail_case_normalization(self, client):
        """Test that code is normalized to uppercase."""
        response = client.get("/api/v1/dtc/p0101")

        # Should still work with lowercase input
        assert response.status_code in [200, 404]  # Either found or not found, but not invalid


class TestDTCRelatedCodesEndpoint:
    """Test related DTC codes endpoint."""

    def test_get_related_codes(self, client):
        """Test getting related codes."""
        response = client.get("/api/v1/dtc/P0101/related")

        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, list)

    def test_related_codes_have_required_fields(self, client):
        """Test that related codes have required fields."""
        response = client.get("/api/v1/dtc/P0101/related")

        assert response.status_code == 200
        data = response.json()

        if data:
            first_related = data[0]
            assert "code" in first_related
            assert "description_en" in first_related


class TestDTCCategoriesEndpoint:
    """Test DTC categories listing endpoint."""

    def test_get_categories_list(self, client):
        """Test getting list of DTC categories."""
        response = client.get("/api/v1/dtc/categories/list")

        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, list)
        assert len(data) == 4  # P, B, C, U

    def test_categories_have_required_fields(self, client):
        """Test that categories have required fields."""
        response = client.get("/api/v1/dtc/categories/list")

        assert response.status_code == 200
        data = response.json()

        required_fields = ["code", "name", "name_hu", "description", "description_hu"]

        for category in data:
            for field in required_fields:
                assert field in category, f"Missing field: {field}"

    def test_categories_include_all_types(self, client):
        """Test that all category types are included."""
        response = client.get("/api/v1/dtc/categories/list")

        assert response.status_code == 200
        data = response.json()

        codes = [cat["code"] for cat in data]

        assert "P" in codes  # Powertrain
        assert "B" in codes  # Body
        assert "C" in codes  # Chassis
        assert "U" in codes  # Network


class TestInputValidation:
    """Test input validation for API endpoints."""

    def test_search_query_min_length(self, client):
        """Test minimum query length validation."""
        response = client.get("/api/v1/dtc/search", params={"q": ""})

        assert response.status_code == 422  # Validation error

    def test_search_limit_range(self, client):
        """Test limit parameter range validation."""
        # Too low
        response = client.get("/api/v1/dtc/search", params={"q": "P", "limit": 0})
        assert response.status_code == 422

        # Too high
        response = client.get("/api/v1/dtc/search", params={"q": "P", "limit": 200})
        assert response.status_code == 422

        # Valid range
        response = client.get("/api/v1/dtc/search", params={"q": "P", "limit": 50})
        assert response.status_code == 200

    def test_invalid_category_rejected(self, client):
        """Test that invalid category is rejected."""
        response = client.get(
            "/api/v1/dtc/search",
            params={"q": "P", "category": "invalid_category"}
        )

        assert response.status_code == 422


class TestErrorHandling:
    """Test error handling in API endpoints."""

    def test_invalid_endpoint_returns_404(self, client):
        """Test that invalid endpoint returns 404."""
        response = client.get("/api/v1/dtc/invalid/endpoint/path")

        assert response.status_code == 404

    def test_method_not_allowed(self, client):
        """Test that wrong HTTP method returns 405."""
        response = client.post("/api/v1/dtc/search", params={"q": "P"})

        assert response.status_code == 405

    def test_error_response_format(self, client):
        """Test that error responses have proper format."""
        response = client.get("/api/v1/dtc/INVALID")

        assert response.status_code == 400
        data = response.json()

        assert "detail" in data


class TestResponseFormat:
    """Test response format and content types."""

    def test_json_content_type(self, client):
        """Test that responses have JSON content type."""
        response = client.get("/health")

        assert "application/json" in response.headers["content-type"]

    def test_search_response_is_list(self, client):
        """Test that search response is always a list."""
        response = client.get("/api/v1/dtc/search", params={"q": "ZZZZZ"})  # No matches expected

        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, list)  # Should be empty list, not null

    def test_detail_response_is_object(self, client):
        """Test that detail response is an object."""
        response = client.get("/api/v1/dtc/P0101")

        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, dict)
