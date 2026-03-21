"""
API tests for vehicle endpoints.

Tests:
- POST /api/v1/vehicles/decode-vin - VIN decoding
- GET /api/v1/vehicles/makes - Get vehicle makes (paginated)
- GET /api/v1/vehicles/models?make=... - Get vehicle models (paginated)
- GET /api/v1/vehicles/years?make=...&model=... - Get available years
- GET /api/v1/vehicles/{make}/{model}/{year}/recalls - Get vehicle recalls
- GET /api/v1/vehicles/{make}/{model}/{year}/complaints - Get vehicle complaints
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import pytest
from unittest.mock import AsyncMock

if TYPE_CHECKING:
    from httpx import AsyncClient

from app.services.nhtsa_service import get_nhtsa_service
from app.services.vehicle_service import get_vehicle_service


class TestVINDecode:
    """Tests for POST /api/v1/vehicles/decode-vin endpoint."""

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
