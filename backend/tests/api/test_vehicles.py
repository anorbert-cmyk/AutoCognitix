"""
API tests for vehicle endpoints.

Tests:
- POST /api/v1/vehicles/decode-vin - VIN decoding
- GET /api/v1/vehicles/makes - Get vehicle makes (paginated)
- GET /api/v1/vehicles/models?make=... - Get vehicle models (paginated)
- GET /api/v1/vehicles/years?make=...&model=... - Get available years
- GET /api/v1/vehicles/{make}/{model}/{year}/recalls - Get vehicle recalls
- GET /api/v1/vehicles/{make}/{model}/{year}/complaints - Get vehicle complaints
- GET /api/v1/vehicles/{make}/{model}/common-issues - DTC ranking + NHTSA
  complaint-component ranking
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from unittest.mock import AsyncMock, MagicMock, patch

if TYPE_CHECKING:
    from httpx import AsyncClient


from app.db.postgres.models import VehicleComplaint
from app.services.nhtsa_service import get_nhtsa_service
from app.services.vehicle_service import get_vehicle_service


class TestVINDecode:
    """Tests for POST /api/v1/vehicles/decode-vin endpoint."""

    @pytest.fixture(autouse=True)
    def _cleanup_overrides(self, app):
        yield
        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_decode_vin_success(
        self, async_client: AsyncClient, app, mock_nhtsa_service, valid_vins: list[str]
    ):
        """Test successful VIN decode returns 200."""
        app.dependency_overrides[get_nhtsa_service] = lambda: mock_nhtsa_service

        response = await async_client.post(
            "/api/v1/vehicles/decode-vin",
            json={"vin": valid_vins[0]},
        )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_decode_vin_returns_vehicle_info(
        self, async_client: AsyncClient, app, mock_nhtsa_service
    ):
        """Test that VIN decode returns vehicle information."""
        app.dependency_overrides[get_nhtsa_service] = lambda: mock_nhtsa_service

        response = await async_client.post(
            "/api/v1/vehicles/decode-vin",
            json={"vin": "WVWZZZ3CZWE123456"},
        )

        assert response.status_code == 200
        data = response.json()

        assert "vin" in data
        assert "make" in data
        assert "model" in data
        assert "year" in data

    @pytest.mark.asyncio
    async def test_decode_vin_normalizes_to_uppercase(
        self, async_client: AsyncClient, app, mock_nhtsa_service
    ):
        """Test that VIN is normalized to uppercase."""
        app.dependency_overrides[get_nhtsa_service] = lambda: mock_nhtsa_service

        response = await async_client.post(
            "/api/v1/vehicles/decode-vin",
            json={"vin": "wvwzzz3czwe123456"},  # lowercase
        )

        assert response.status_code == 200
        data = response.json()
        assert data["vin"] == "WVWZZZ3CZWE123456"

    @pytest.mark.asyncio
    async def test_decode_vin_too_short_returns_422(self, async_client: AsyncClient):
        """Test that VIN under 17 characters returns 422 (Pydantic validation)."""
        response = await async_client.post(
            "/api/v1/vehicles/decode-vin",
            json={"vin": "WVWZZZ3CZWE"},  # 11 chars
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_decode_vin_too_long_returns_422(self, async_client: AsyncClient):
        """Test that VIN over 17 characters returns 422 (Pydantic validation)."""
        response = await async_client.post(
            "/api/v1/vehicles/decode-vin",
            json={"vin": "WVWZZZ3CZWE123456789"},  # 20 chars
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_decode_vin_with_invalid_char_i_returns_400(self, async_client: AsyncClient):
        """Test that VIN containing 'I' returns 400."""
        response = await async_client.post(
            "/api/v1/vehicles/decode-vin",
            json={"vin": "WVWZZZ3CZWE12345I"},  # Contains I - but only 17 chars
        )

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_decode_vin_with_invalid_char_o_returns_400(self, async_client: AsyncClient):
        """Test that VIN containing 'O' returns 400."""
        response = await async_client.post(
            "/api/v1/vehicles/decode-vin",
            json={"vin": "WVWZZZ3CZWE12345O"},  # Contains O - but only 17 chars
        )

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_decode_vin_with_invalid_char_q_returns_400(self, async_client: AsyncClient):
        """Test that VIN containing 'Q' returns 400."""
        response = await async_client.post(
            "/api/v1/vehicles/decode-vin",
            json={"vin": "WVWZZZ3CZWE12345Q"},  # Contains Q - but only 17 chars
        )

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_decode_vin_missing_vin_returns_422(self, async_client: AsyncClient):
        """Test that missing VIN returns 422."""
        response = await async_client.post(
            "/api/v1/vehicles/decode-vin",
            json={},
        )

        assert response.status_code == 422


class TestVehicleMakes:
    """Tests for GET /api/v1/vehicles/makes endpoint (paginated)."""

    @pytest.fixture(autouse=True)
    def _cleanup_overrides(self, app):
        yield
        app.dependency_overrides.clear()

    @pytest.fixture
    def mock_vehicle_svc(self):
        """Create a mock vehicle service with makes data."""
        mock = AsyncMock()
        mock.get_all_makes.return_value = (
            [
                {"id": "volkswagen", "name": "Volkswagen", "country": "Germany"},
                {"id": "toyota", "name": "Toyota", "country": "Japan"},
                {"id": "bmw", "name": "BMW", "country": "Germany"},
                {"id": "ford", "name": "Ford", "country": "USA"},
                {"id": "audi", "name": "Audi", "country": "Germany"},
            ],
            5,
        )
        return mock

    @pytest.mark.asyncio
    async def test_get_makes_returns_200(self, async_client: AsyncClient, app, mock_vehicle_svc):
        """Test getting vehicle makes returns 200."""
        app.dependency_overrides[get_vehicle_service] = lambda: mock_vehicle_svc

        response = await async_client.get("/api/v1/vehicles/makes")

        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert isinstance(data["items"], list)

    @pytest.mark.asyncio
    async def test_get_makes_returns_make_objects(
        self, async_client: AsyncClient, app, mock_vehicle_svc
    ):
        """Test that makes have correct structure."""
        app.dependency_overrides[get_vehicle_service] = lambda: mock_vehicle_svc

        response = await async_client.get("/api/v1/vehicles/makes")

        assert response.status_code == 200
        data = response.json()

        assert len(data["items"]) > 0
        make = data["items"][0]
        assert "id" in make
        assert "name" in make
        assert "country" in make

    @pytest.mark.asyncio
    async def test_get_makes_includes_common_brands(
        self, async_client: AsyncClient, app, mock_vehicle_svc
    ):
        """Test that common car brands are included."""
        app.dependency_overrides[get_vehicle_service] = lambda: mock_vehicle_svc

        response = await async_client.get("/api/v1/vehicles/makes")

        assert response.status_code == 200
        data = response.json()

        make_names = [m["name"].lower() for m in data["items"]]
        assert "volkswagen" in make_names
        assert "toyota" in make_names
        assert "bmw" in make_names

    @pytest.mark.asyncio
    async def test_get_makes_with_search_filter(self, async_client: AsyncClient, app):
        """Test filtering makes by search term."""
        mock = AsyncMock()
        mock.get_all_makes.return_value = (
            [{"id": "volkswagen", "name": "Volkswagen", "country": "Germany"}],
            1,
        )
        app.dependency_overrides[get_vehicle_service] = lambda: mock

        response = await async_client.get(
            "/api/v1/vehicles/makes",
            params={"search": "volk"},
        )

        assert response.status_code == 200
        data = response.json()

        # All results should contain "volk"
        for make in data["items"]:
            assert "volk" in make["name"].lower()

    @pytest.mark.asyncio
    async def test_get_makes_search_case_insensitive(self, async_client: AsyncClient, app):
        """Test that make search is case insensitive."""
        mock = AsyncMock()
        mock.get_all_makes.return_value = (
            [{"id": "volkswagen", "name": "Volkswagen", "country": "Germany"}],
            1,
        )
        app.dependency_overrides[get_vehicle_service] = lambda: mock

        response_lower = await async_client.get(
            "/api/v1/vehicles/makes",
            params={"search": "volk"},
        )
        response_upper = await async_client.get(
            "/api/v1/vehicles/makes",
            params={"search": "VOLK"},
        )

        assert response_lower.status_code == 200
        assert response_upper.status_code == 200

        data_lower = response_lower.json()
        data_upper = response_upper.json()

        assert len(data_lower["items"]) == len(data_upper["items"])

    @pytest.mark.asyncio
    async def test_get_makes_search_no_results(self, async_client: AsyncClient, app):
        """Test that search with no matches returns empty list."""
        mock = AsyncMock()
        mock.get_all_makes.return_value = ([], 0)
        app.dependency_overrides[get_vehicle_service] = lambda: mock

        response = await async_client.get(
            "/api/v1/vehicles/makes",
            params={"search": "zzzznonexistent"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["items"] == []


class TestVehicleModels:
    """Tests for GET /api/v1/vehicles/models?make=... endpoint (paginated)."""

    @pytest.fixture(autouse=True)
    def _cleanup_overrides(self, app):
        yield
        app.dependency_overrides.clear()

    @pytest.fixture
    def mock_vehicle_svc_models(self):
        """Create a mock vehicle service with models data."""
        mock = AsyncMock()
        mock.get_models_for_make.return_value = (
            [
                {
                    "id": "golf",
                    "name": "Golf",
                    "make_id": "volkswagen",
                    "year_start": 1974,
                    "year_end": None,
                    "body_types": ["Hatchback"],
                },
                {
                    "id": "passat",
                    "name": "Passat",
                    "make_id": "volkswagen",
                    "year_start": 1973,
                    "year_end": None,
                    "body_types": ["Sedan"],
                },
            ],
            2,
        )
        return mock

    @pytest.mark.asyncio
    async def test_get_models_returns_200(
        self, async_client: AsyncClient, app, mock_vehicle_svc_models
    ):
        """Test getting vehicle models returns 200."""
        app.dependency_overrides[get_vehicle_service] = lambda: mock_vehicle_svc_models

        response = await async_client.get(
            "/api/v1/vehicles/models",
            params={"make": "volkswagen"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert isinstance(data["items"], list)

    @pytest.mark.asyncio
    async def test_get_models_returns_model_objects(
        self, async_client: AsyncClient, app, mock_vehicle_svc_models
    ):
        """Test that models have correct structure."""
        app.dependency_overrides[get_vehicle_service] = lambda: mock_vehicle_svc_models

        response = await async_client.get(
            "/api/v1/vehicles/models",
            params={"make": "volkswagen"},
        )

        assert response.status_code == 200
        data = response.json()

        if data["items"]:
            model = data["items"][0]
            assert "id" in model
            assert "name" in model
            assert "make_id" in model
            assert "year_start" in model

    @pytest.mark.asyncio
    async def test_get_models_includes_common_models(
        self, async_client: AsyncClient, app, mock_vehicle_svc_models
    ):
        """Test that common VW models are included."""
        app.dependency_overrides[get_vehicle_service] = lambda: mock_vehicle_svc_models

        response = await async_client.get(
            "/api/v1/vehicles/models",
            params={"make": "volkswagen"},
        )

        assert response.status_code == 200
        data = response.json()

        model_names = [m["name"].lower() for m in data["items"]]
        assert "golf" in model_names or len(data["items"]) == 0

    @pytest.mark.asyncio
    async def test_get_models_with_year_filter(self, async_client: AsyncClient, app):
        """Test filtering models by search (year filter is done client-side, test search param)."""
        mock = AsyncMock()
        mock.get_models_for_make.return_value = (
            [
                {
                    "id": "golf",
                    "name": "Golf",
                    "make_id": "volkswagen",
                    "year_start": 1974,
                    "year_end": None,
                    "body_types": ["Hatchback"],
                },
            ],
            1,
        )
        app.dependency_overrides[get_vehicle_service] = lambda: mock

        response = await async_client.get(
            "/api/v1/vehicles/models",
            params={"make": "volkswagen", "search": "golf"},
        )

        assert response.status_code == 200
        data = response.json()

        for model in data["items"]:
            assert model["year_start"] <= 2020

    @pytest.mark.asyncio
    async def test_get_models_unknown_make_returns_404(self, async_client: AsyncClient, app):
        """Test that unknown make returns 404."""
        mock = AsyncMock()
        mock.get_models_for_make.return_value = ([], 0)
        app.dependency_overrides[get_vehicle_service] = lambda: mock

        response = await async_client.get(
            "/api/v1/vehicles/models",
            params={"make": "nonexistent_make"},
        )

        # The endpoint returns 404 when no models found with offset=0
        assert response.status_code == 404


class TestVehicleYears:
    """Tests for GET /api/v1/vehicles/years?make=...&model=... endpoint."""

    @pytest.fixture(autouse=True)
    def _cleanup_overrides(self, app):
        yield
        app.dependency_overrides.clear()

    @pytest.fixture
    def mock_vehicle_svc_years(self):
        """Create a mock vehicle service with years data."""
        from datetime import datetime as dt

        current_year = dt.now().year
        next_year = current_year + 1

        mock = AsyncMock()
        # Descending order, from next year down to 1980
        years = list(range(next_year, 1979, -1))
        mock.get_years_for_vehicle.return_value = years
        return mock

    @pytest.mark.asyncio
    async def test_get_years_returns_200(
        self, async_client: AsyncClient, app, mock_vehicle_svc_years
    ):
        """Test getting years returns 200."""
        app.dependency_overrides[get_vehicle_service] = lambda: mock_vehicle_svc_years

        response = await async_client.get(
            "/api/v1/vehicles/years",
            params={"make": "Volkswagen", "model": "Golf"},
        )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_get_years_returns_years_list(
        self, async_client: AsyncClient, app, mock_vehicle_svc_years
    ):
        """Test that years response includes years list."""
        app.dependency_overrides[get_vehicle_service] = lambda: mock_vehicle_svc_years

        response = await async_client.get(
            "/api/v1/vehicles/years",
            params={"make": "Volkswagen", "model": "Golf"},
        )

        assert response.status_code == 200
        data = response.json()

        assert "years" in data
        assert isinstance(data["years"], list)

    @pytest.mark.asyncio
    async def test_get_years_includes_current_year(
        self, async_client: AsyncClient, app, mock_vehicle_svc_years
    ):
        """Test that current year is included."""
        from datetime import datetime

        app.dependency_overrides[get_vehicle_service] = lambda: mock_vehicle_svc_years

        response = await async_client.get(
            "/api/v1/vehicles/years",
            params={"make": "Volkswagen", "model": "Golf"},
        )

        assert response.status_code == 200
        data = response.json()

        current_year = datetime.now().year
        assert current_year in data["years"]

    @pytest.mark.asyncio
    async def test_get_years_includes_next_year(
        self, async_client: AsyncClient, app, mock_vehicle_svc_years
    ):
        """Test that next model year is included."""
        from datetime import datetime

        app.dependency_overrides[get_vehicle_service] = lambda: mock_vehicle_svc_years

        response = await async_client.get(
            "/api/v1/vehicles/years",
            params={"make": "Volkswagen", "model": "Golf"},
        )

        assert response.status_code == 200
        data = response.json()

        next_year = datetime.now().year + 1
        assert next_year in data["years"]

    @pytest.mark.asyncio
    async def test_get_years_sorted_descending(
        self, async_client: AsyncClient, app, mock_vehicle_svc_years
    ):
        """Test that years are sorted in descending order."""
        app.dependency_overrides[get_vehicle_service] = lambda: mock_vehicle_svc_years

        response = await async_client.get(
            "/api/v1/vehicles/years",
            params={"make": "Volkswagen", "model": "Golf"},
        )

        assert response.status_code == 200
        data = response.json()

        years = data["years"]
        assert years == sorted(years, reverse=True)

    @pytest.mark.asyncio
    async def test_get_years_starts_from_1980(
        self, async_client: AsyncClient, app, mock_vehicle_svc_years
    ):
        """Test that years go back to 1980."""
        app.dependency_overrides[get_vehicle_service] = lambda: mock_vehicle_svc_years

        response = await async_client.get(
            "/api/v1/vehicles/years",
            params={"make": "Volkswagen", "model": "Golf"},
        )

        assert response.status_code == 200
        data = response.json()

        assert 1980 in data["years"]


class TestVehicleRecalls:
    """Tests for GET /api/v1/vehicles/{make}/{model}/{year}/recalls endpoint."""

    @pytest.fixture(autouse=True)
    def _cleanup_overrides(self, app):
        yield
        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_get_recalls_returns_200(
        self, async_client: AsyncClient, app, mock_nhtsa_service
    ):
        """Test getting recalls returns 200."""
        app.dependency_overrides[get_nhtsa_service] = lambda: mock_nhtsa_service

        response = await async_client.get("/api/v1/vehicles/Volkswagen/Golf/2018/recalls")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_get_recalls_returns_recall_objects(
        self, async_client: AsyncClient, app, mock_nhtsa_service
    ):
        """Test that recalls have correct structure."""
        app.dependency_overrides[get_nhtsa_service] = lambda: mock_nhtsa_service

        response = await async_client.get("/api/v1/vehicles/Volkswagen/Golf/2018/recalls")

        assert response.status_code == 200
        data = response.json()

        if data:
            recall = data[0]
            assert "campaign_number" in recall
            assert "manufacturer" in recall
            assert "summary" in recall

    @pytest.mark.asyncio
    async def test_get_recalls_invalid_year_returns_422(self, async_client: AsyncClient):
        """Test that invalid year returns 422."""
        response = await async_client.get(
            "/api/v1/vehicles/Volkswagen/Golf/1899/recalls"  # Before 1900
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_get_recalls_future_year_returns_422(self, async_client: AsyncClient):
        """Test that future year beyond limit returns 422."""
        response = await async_client.get(
            "/api/v1/vehicles/Volkswagen/Golf/2050/recalls"  # Beyond 2030
        )

        assert response.status_code == 422


class TestVehicleComplaints:
    """Tests for GET /api/v1/vehicles/{make}/{model}/{year}/complaints endpoint."""

    @pytest.fixture(autouse=True)
    def _cleanup_overrides(self, app):
        yield
        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_get_complaints_returns_200(
        self, async_client: AsyncClient, app, mock_nhtsa_service
    ):
        """Test getting complaints returns 200."""
        app.dependency_overrides[get_nhtsa_service] = lambda: mock_nhtsa_service

        response = await async_client.get("/api/v1/vehicles/Volkswagen/Golf/2018/complaints")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_get_complaints_returns_complaint_objects(
        self, async_client: AsyncClient, app, mock_nhtsa_service
    ):
        """Test that complaints have correct structure."""
        app.dependency_overrides[get_nhtsa_service] = lambda: mock_nhtsa_service

        response = await async_client.get("/api/v1/vehicles/Volkswagen/Golf/2018/complaints")

        assert response.status_code == 200
        data = response.json()

        if data:
            complaint = data[0]
            assert "manufacturer" in complaint
            assert "summary" in complaint

    @pytest.mark.asyncio
    async def test_get_complaints_invalid_year_returns_422(self, async_client: AsyncClient):
        """Test that invalid year returns 422."""
        response = await async_client.get(
            "/api/v1/vehicles/Volkswagen/Golf/1899/complaints"  # Before 1900
        )

        assert response.status_code == 422


class TestVehicleEndpointErrors:
    """Tests for vehicle endpoint error handling."""

    @pytest.fixture(autouse=True)
    def _cleanup_overrides(self, app):
        yield
        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_nhtsa_api_error_returns_502(self, async_client: AsyncClient, app):
        """Test that NHTSA API errors return 502."""
        from app.services.nhtsa_service import NHTSAError

        mock_service = AsyncMock()
        mock_service.decode_vin.side_effect = NHTSAError("Service unavailable")

        app.dependency_overrides[get_nhtsa_service] = lambda: mock_service

        response = await async_client.post(
            "/api/v1/vehicles/decode-vin",
            json={"vin": "WVWZZZ3CZWE123456"},
        )

        assert response.status_code == 502

    @pytest.mark.asyncio
    async def test_recalls_nhtsa_error_returns_502(self, async_client: AsyncClient, app):
        """Test that NHTSA API errors for recalls return 502."""
        from app.services.nhtsa_service import NHTSAError

        mock_service = AsyncMock()
        mock_service.get_recalls.side_effect = NHTSAError("Service unavailable")

        app.dependency_overrides[get_nhtsa_service] = lambda: mock_service

        response = await async_client.get("/api/v1/vehicles/Volkswagen/Golf/2018/recalls")

        assert response.status_code == 502

    @pytest.mark.asyncio
    async def test_complaints_nhtsa_error_returns_502(self, async_client: AsyncClient, app):
        """Test that NHTSA API errors for complaints return 502."""
        from app.services.nhtsa_service import NHTSAError

        mock_service = AsyncMock()
        mock_service.get_complaints.side_effect = NHTSAError("Service unavailable")

        app.dependency_overrides[get_nhtsa_service] = lambda: mock_service

        response = await async_client.get("/api/v1/vehicles/Volkswagen/Golf/2018/complaints")

        assert response.status_code == 502


class TestVehicleCommonIssues:
    """Tests for GET /api/v1/vehicles/{make}/{model}/common-issues endpoint.

    Regression coverage for the production 500 (dead Neo4j label path). The
    endpoint must return a ranked list on data and 200-with-empty otherwise.
    """

    @pytest.fixture(autouse=True)
    def _cleanup_overrides(self, app):
        yield
        app.dependency_overrides.clear()

    @pytest.fixture(autouse=True)
    def _no_component_query(self):
        """Neutralise the PostgreSQL component aggregation for the DTC-path tests.

        `async_session_maker` is module-level (never dependency-overridden) and
        points at a PostgreSQL that does not exist in the test harness, so the
        tests below that exercise the REAL VehicleService would otherwise fail on
        an unrelated connection error. Patched to a healthy-but-empty result so
        those tests keep asserting exactly what they were written to assert.
        `TestVehicleComplaintComponents` covers the component path for real.
        """
        empty_result = MagicMock()
        empty_result.mappings.return_value.all.return_value = []

        session = AsyncMock()
        session.execute = AsyncMock(return_value=empty_result)

        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=session)
        ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("app.services.vehicle_service.async_session_maker", return_value=ctx):
            yield

    @pytest.mark.asyncio
    async def test_common_issues_returns_200_ranked(self, async_client: AsyncClient, app):
        mock = AsyncMock()
        mock.get_vehicle_complaint_components.return_value = ([], 0)
        mock.get_vehicle_common_issues.return_value = [
            {
                "code": "P0301",
                "description_en": "Cylinder 1 Misfire Detected",
                "description_hu": "1. henger gyujtaskihagyas",
                "severity": "high",
                "frequency": "very_common",
                "occurrence_count": 42,
            },
            {
                "code": "P0420",
                "description_en": "Catalyst System Efficiency Below Threshold",
                "description_hu": "Katalizator hatekonysag a kuszob alatt",
                "severity": "medium",
                "frequency": "common",
                "occurrence_count": 7,
            },
        ]
        app.dependency_overrides[get_vehicle_service] = lambda: mock

        response = await async_client.get(
            "/api/v1/vehicles/Volkswagen/Golf/common-issues?year=2018"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["make"] == "Volkswagen"
        assert data["model"] == "Golf"
        assert data["year"] == 2018
        assert [i["code"] for i in data["issues"]] == ["P0301", "P0420"]
        assert data["issues"][0]["occurrence_count"] == 42
        mock.get_vehicle_common_issues.assert_awaited_once_with(
            make="Volkswagen", model="Golf", year=2018
        )

    @pytest.mark.asyncio
    async def test_common_issues_empty_returns_200_not_500(self, async_client: AsyncClient, app):
        """No graph data must yield 200 with empty issues, never 500."""
        mock = AsyncMock()
        mock.get_vehicle_common_issues.return_value = []
        mock.get_vehicle_complaint_components.return_value = ([], 0)
        app.dependency_overrides[get_vehicle_service] = lambda: mock

        response = await async_client.get("/api/v1/vehicles/Toyota/Corolla/common-issues")

        assert response.status_code == 200
        data = response.json()
        assert data["issues"] == []
        assert data["make"] == "Toyota"
        assert data["year"] is None

    @pytest.mark.asyncio
    async def test_common_issues_without_year(self, async_client: AsyncClient, app):
        mock = AsyncMock()
        mock.get_vehicle_complaint_components.return_value = ([], 0)
        mock.get_vehicle_common_issues.return_value = [
            {
                "code": "P0171",
                "description_en": "System Too Lean",
                "description_hu": "Rendszer tul sovany",
                "severity": "medium",
                "frequency": "common",
                "occurrence_count": 10,
            },
        ]
        app.dependency_overrides[get_vehicle_service] = lambda: mock

        response = await async_client.get("/api/v1/vehicles/Volkswagen/Golf/common-issues")

        assert response.status_code == 200
        data = response.json()
        assert data["year"] is None
        assert data["issues"][0]["code"] == "P0171"

    @pytest.mark.asyncio
    async def test_common_issues_neo4j_outage_returns_200_and_logs_error(
        self, async_client: AsyncClient, caplog
    ):
        """A Neo4j outage keeps the 200/[] contract but is loud in the logs.

        Uses the real VehicleService (no dependency override) so the endpoint's
        externally visible behaviour and the service's logging are checked together.
        """
        with (
            caplog.at_level(logging.INFO, logger="app.services.vehicle_service"),
            patch("asyncio.to_thread", side_effect=RuntimeError("ServiceUnavailable")),
        ):
            response = await async_client.get(
                "/api/v1/vehicles/Volkswagen/Golf/common-issues?year=2018"
            )

        assert response.status_code == 200
        assert response.json()["issues"] == []

        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1
        assert "common-issues neo4j QUERY FAILED" in errors[0].getMessage()

    @pytest.mark.asyncio
    async def test_common_issues_genuine_empty_logs_no_error(
        self, async_client: AsyncClient, caplog
    ):
        """A healthy graph with no matching rows returns the same 200/[] but must
        NOT log an error - that is what makes the outage above diagnosable.
        """

        async def _to_thread(fn, *args, **kwargs):
            return [], None

        with (
            caplog.at_level(logging.INFO, logger="app.services.vehicle_service"),
            patch("asyncio.to_thread", side_effect=_to_thread),
        ):
            response = await async_client.get("/api/v1/vehicles/Toyota/Corolla/common-issues")

        assert response.status_code == 200
        assert response.json()["issues"] == []
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("common-issues neo4j OK" in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_sources_report_ok_when_both_datastores_answer(
        self, async_client: AsyncClient, app
    ):
        """Empty AND `ok` is the ONLY combination a client may present as an
        absence of data ("no NHTSA record for this vehicle").
        """
        mock = AsyncMock()
        mock.get_vehicle_common_issues.return_value = []
        mock.get_vehicle_complaint_components.return_value = ([], 0)
        app.dependency_overrides[get_vehicle_service] = lambda: mock

        response = await async_client.get("/api/v1/vehicles/Skoda/Octavia/common-issues")

        assert response.status_code == 200
        assert response.json()["sources"] == {"components": "ok", "issues": "ok"}

    @pytest.mark.asyncio
    async def test_sources_flag_the_graph_unavailable_on_a_swallowed_neo4j_error(
        self, async_client: AsyncClient
    ):
        """The swallowed outage must be visible to the CLIENT, not just in the logs.

        Uses the real VehicleService so the swallow and the flag are checked
        together. The empty `issues` list is unchanged (the 200 contract), but it
        is now labelled as "could not load" rather than "nothing to report".
        """
        with patch("asyncio.to_thread", side_effect=RuntimeError("ServiceUnavailable")):
            response = await async_client.get("/api/v1/vehicles/Volkswagen/Golf/common-issues")

        assert response.status_code == 200
        data = response.json()
        assert data["issues"] == []
        assert data["sources"]["issues"] == "unavailable"
        # The sibling source is independent and must not be tarred with it
        assert data["sources"]["components"] == "ok"

    @pytest.mark.asyncio
    async def test_whitespace_only_param_is_rejected(self, async_client: AsyncClient, app):
        """A whitespace-only make/model is refused instead of matching everything.

        `_VEHICLE_PARAM_RE` accepts a lone space, which strips to "" - and an
        empty model is not a narrower filter but NO filter (SQL `LIKE '%'`,
        Cypher `'golf gti' STARTS WITH ''`). Rejected once in the endpoint, so
        BOTH legs are covered; the service is never even called.
        """
        mock = AsyncMock()
        mock.get_vehicle_common_issues.return_value = []
        mock.get_vehicle_complaint_components.return_value = ([], 0)
        app.dependency_overrides[get_vehicle_service] = lambda: mock

        for path in (
            "/api/v1/vehicles/Volkswagen/%20/common-issues",
            "/api/v1/vehicles/Volkswagen/%20%20/common-issues",
            "/api/v1/vehicles/%20/Golf/common-issues",
        ):
            response = await async_client.get(path)
            assert response.status_code == 422, path
            assert response.json()["detail"] == "A gyártó és a modell nem lehet üres."

        mock.get_vehicle_common_issues.assert_not_awaited()
        mock.get_vehicle_complaint_components.assert_not_awaited()


class TestVehicleComplaintComponents:
    """The `components` half of GET /{make}/{model}/common-issues.

    Ranks a vehicle's components by NHTSA consumer-complaint frequency out of
    PostgreSQL. Unlike the DTC ranking - which needs a complaint narrative to
    literally quote a fault code, something consumers essentially never do -
    this list has real coverage, so these tests drive the ACTUAL aggregation SQL
    against a seeded database rather than a mocked service.
    """

    @pytest.fixture(autouse=True)
    def _cleanup_overrides(self, app):
        yield
        app.dependency_overrides.clear()

    @pytest.fixture(autouse=True)
    def _no_neo4j(self):
        """The DTC path is not under test here; keep it a healthy empty graph."""

        async def _to_thread(fn, *args, **kwargs):
            return [], None

        with patch("asyncio.to_thread", side_effect=_to_thread):
            yield

    @pytest_asyncio.fixture
    async def seeded_complaints(self, async_engine):
        """Seed `vehicle_complaints` and point the service's session maker at it.

        `async_session_maker` is module-level (never dependency-overridden), so
        binding it to the test engine is what lets the real aggregation SQL run.
        Values mirror the real corpus: UPPERCASE make/model, trim-qualified model
        names, and NHTSA's own uppercase component labels.
        """
        session_factory = async_sessionmaker(
            async_engine, class_=AsyncSession, expire_on_commit=False
        )

        rows = [
            # (odi, make, model, year, crash, fire, injuries, deaths, component)
            ("1", "VOLKSWAGEN", "GOLF", 2018, True, False, 2, 0, "ELECTRICAL SYSTEM"),
            ("2", "VOLKSWAGEN", "GOLF", 2018, False, True, 0, 1, "ELECTRICAL SYSTEM"),
            ("3", "VOLKSWAGEN", "GOLF GTI", 2018, False, False, 0, 0, "ELECTRICAL SYSTEM"),
            ("4", "VOLKSWAGEN", "GOLF R", 2018, False, False, 0, 0, "ENGINE"),
            ("5", "VOLKSWAGEN", "GOLF SPORTWAGEN", 2018, False, False, 0, 0, "SOME WIDGET"),
            ("6", "VOLKSWAGEN", "GOLF", 2019, False, False, 0, 0, "ENGINE"),
            # Different model under the same make - must NOT leak into GOLF
            ("7", "VOLKSWAGEN", "JETTA", 2018, False, False, 0, 0, "STEERING"),
            # Different make - must NOT leak in either
            ("8", "AUDI", "GOLF CART", 2018, False, False, 0, 0, "STEERING"),
            # NHTSA ships BOTH spellings of this brand as separate makes
            ("9", "MERCEDES-BENZ", "GLC-CLASS", 2018, False, False, 0, 0, "SUSPENSION"),
            ("10", "MERCEDES BENZ", "GLC-CLASS COUPE", 2018, False, False, 0, 0, "SUSPENSION"),
            # Blank component folds into the corpus's own UNKNOWN bucket
            ("11", "MERCEDES-BENZ", "GLC-CLASS", 2018, False, False, 0, 0, None),
        ]

        async with session_factory() as session:
            for odi, make, model, year, crash, fire, inj, dead, component in rows:
                session.add(
                    VehicleComplaint(
                        odi_number=odi,
                        manufacturer=make,
                        make=make,
                        model=model,
                        model_year=year,
                        crash=crash,
                        fire=fire,
                        injuries=inj,
                        deaths=dead,
                        components=component,
                        extracted_dtc_codes=[],
                    )
                )
            await session.commit()

        with patch(
            "app.services.vehicle_service.async_session_maker",
            side_effect=session_factory,
        ):
            yield

    @pytest.mark.asyncio
    async def test_components_ranked_with_counts_shares_and_safety(
        self, async_client: AsyncClient, seeded_complaints
    ):
        """Case-insensitive make + prefix model match, ranked by count DESC."""
        response = await async_client.get(
            "/api/v1/vehicles/Volkswagen/Golf/common-issues?year=2018"
        )

        assert response.status_code == 200
        data = response.json()

        # 5 GOLF* rows in 2018 (GOLF x2, GOLF GTI, GOLF R, GOLF SPORTWAGEN);
        # JETTA and the AUDI row are correctly excluded.
        assert data["total_complaints"] == 5
        assert [c["component"] for c in data["components"]] == [
            "ELECTRICAL SYSTEM",
            "ENGINE",
            "SOME WIDGET",
        ]

        electrical = data["components"][0]
        assert electrical["complaint_count"] == 3
        assert electrical["share"] == 0.6
        assert electrical["crash_count"] == 1
        assert electrical["fire_count"] == 1
        assert electrical["injury_count"] == 2
        assert electrical["death_count"] == 1
        assert electrical["component_hu"] == "Elektromos rendszer"

        # Deterministic tiebreak: equal counts sort by component name ASC
        assert data["components"][1]["complaint_count"] == 1
        assert data["components"][2]["complaint_count"] == 1
        # Unmapped component -> null, never an invented translation
        assert data["components"][2]["component_hu"] is None

    @pytest.mark.asyncio
    async def test_components_without_year_covers_all_years(
        self, async_client: AsyncClient, seeded_complaints
    ):
        response = await async_client.get("/api/v1/vehicles/volkswagen/GOLF/common-issues")

        assert response.status_code == 200
        data = response.json()
        # 2019 GOLF row is now included
        assert data["total_complaints"] == 6
        counts = {c["component"]: c["complaint_count"] for c in data["components"]}
        assert counts == {"ELECTRICAL SYSTEM": 3, "ENGINE": 2, "SOME WIDGET": 1}
        assert data["components"][0]["share"] == 0.5

    @pytest.mark.asyncio
    async def test_make_spelling_variants_are_both_matched(
        self, async_client: AsyncClient, seeded_complaints
    ):
        """NHTSA stores MERCEDES-BENZ and MERCEDES BENZ as separate makes; a single
        equality would silently drop a large slice of the brand's complaints.
        Also covers the blank-component -> UNKNOWN fold.
        """
        response = await async_client.get(
            "/api/v1/vehicles/Mercedes-Benz/GLC/common-issues?year=2018"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_complaints"] == 3
        counts = {c["component"]: c["complaint_count"] for c in data["components"]}
        assert counts == {"SUSPENSION": 2, "UNKNOWN": 1}
        assert data["components"][0]["component_hu"] == "Futómű"

    @pytest.mark.asyncio
    async def test_limit_caps_components_but_not_the_total(
        self, async_client: AsyncClient, seeded_complaints
    ):
        """`total_complaints` is the pre-LIMIT denominator, so `share` stays honest
        even when the component list is truncated.
        """
        response = await async_client.get(
            "/api/v1/vehicles/Volkswagen/Golf/common-issues?year=2018&limit=1"
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["components"]) == 1
        assert data["total_complaints"] == 5
        assert data["components"][0]["share"] == 0.6

    @pytest.mark.asyncio
    async def test_limit_out_of_range_is_rejected(
        self, async_client: AsyncClient, seeded_complaints
    ):
        assert (
            await async_client.get("/api/v1/vehicles/Volkswagen/Golf/common-issues?limit=0")
        ).status_code == 422
        assert (
            await async_client.get("/api/v1/vehicles/Volkswagen/Golf/common-issues?limit=51")
        ).status_code == 422

    @pytest.mark.asyncio
    async def test_zero_complaint_vehicle_returns_200_empty_and_zero_total(
        self, async_client: AsyncClient, seeded_complaints, caplog
    ):
        """Skoda is one of the 20 seeded European makes with no NHTSA presence at
        all (never sold in the US). That must read as a truthful empty, not an
        error - and it must be logged at INFO so it stays distinguishable from an
        outage.
        """
        with caplog.at_level(logging.INFO, logger="app.services.vehicle_service"):
            response = await async_client.get("/api/v1/vehicles/Skoda/Octavia/common-issues")

        assert response.status_code == 200
        data = response.json()
        assert data["components"] == []
        assert data["total_complaints"] == 0
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("common-issues components OK" in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_db_error_returns_200_with_error_log_not_500(
        self, async_client: AsyncClient, caplog
    ):
        """A PostgreSQL outage keeps the 200 contract with an empty component list,
        and is loud in the logs.
        """
        broken = AsyncMock()
        broken.__aenter__ = AsyncMock(side_effect=RuntimeError("PG down"))
        broken.__aexit__ = AsyncMock(return_value=False)

        with (
            caplog.at_level(logging.INFO, logger="app.services.vehicle_service"),
            patch("app.services.vehicle_service.async_session_maker", return_value=broken),
        ):
            response = await async_client.get(
                "/api/v1/vehicles/Volkswagen/Golf/common-issues?year=2018"
            )

        assert response.status_code == 200
        data = response.json()
        assert data["components"] == []
        assert data["total_complaints"] == 0

        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1
        assert "common-issues components QUERY FAILED" in errors[0].getMessage()
        # ...and the client is told, instead of having to read our logs
        assert data["sources"] == {"components": "unavailable", "issues": "ok"}

    @pytest.mark.asyncio
    async def test_whitespace_only_model_does_not_aggregate_the_whole_make(
        self, async_client: AsyncClient, seeded_complaints
    ):
        """The concrete D2 failure, against real seeded rows.

        Unguarded, `/Volkswagen/%20/common-issues` compiled to `LIKE '%'` and
        returned EVERY Volkswagen component - JETTA's STEERING complaint
        included, even though the caller asked about a (blank) model. The
        request must be refused, not answered with the make's whole history.
        """
        # Control: the make really does have rows that could leak, and one of
        # them belongs to a different model.
        control = await async_client.get("/api/v1/vehicles/Volkswagen/Jetta/common-issues")
        assert control.status_code == 200
        assert [c["component"] for c in control.json()["components"]] == ["STEERING"]

        response = await async_client.get("/api/v1/vehicles/Volkswagen/%20/common-issues")

        assert response.status_code == 422
        body = response.json()
        assert "components" not in body
        assert "STEERING" not in response.text

    @pytest.mark.asyncio
    async def test_legacy_issues_field_still_present(
        self, async_client: AsyncClient, app, seeded_complaints
    ):
        """Additive contract guard: the DTC `issues` list must survive alongside
        the `components` / `total_complaints` / `sources` fields.
        """
        mock = AsyncMock()
        mock.get_vehicle_common_issues.return_value = [
            {
                "code": "P0301",
                "description_en": "Cylinder 1 Misfire Detected",
                "description_hu": "1. henger gyujtaskihagyas",
                "severity": "high",
                "frequency": "very_common",
                "occurrence_count": 42,
            }
        ]
        mock.get_vehicle_complaint_components.return_value = (
            [
                {
                    "component": "ELECTRICAL SYSTEM",
                    "component_hu": "Elektromos rendszer",
                    "complaint_count": 3,
                    "share": 0.6,
                    "crash_count": 1,
                    "fire_count": 1,
                    "injury_count": 2,
                    "death_count": 1,
                }
            ],
            5,
        )
        app.dependency_overrides[get_vehicle_service] = lambda: mock

        response = await async_client.get(
            "/api/v1/vehicles/Volkswagen/Golf/common-issues?year=2018&limit=10"
        )

        assert response.status_code == 200
        data = response.json()
        # `sources` is the additive field from the truthfulness fix; the exact-set
        # assertion is the point of this test, so it has to name it.
        assert set(data) == {
            "make",
            "model",
            "year",
            "issues",
            "components",
            "total_complaints",
            "sources",
        }
        assert data["sources"] == {"components": "ok", "issues": "ok"}
        assert [i["code"] for i in data["issues"]] == ["P0301"]
        assert data["issues"][0]["occurrence_count"] == 42
        assert data["components"][0]["component"] == "ELECTRICAL SYSTEM"
        assert data["total_complaints"] == 5
        # The legacy DTC call keeps its exact signature (no `limit` leaking in)
        mock.get_vehicle_common_issues.assert_awaited_once_with(
            make="Volkswagen", model="Golf", year=2018
        )
        mock.get_vehicle_complaint_components.assert_awaited_once_with(
            make="Volkswagen", model="Golf", year=2018, limit=10
        )
