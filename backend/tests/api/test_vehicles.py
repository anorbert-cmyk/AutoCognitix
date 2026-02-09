"""
API tests for vehicle endpoints.

Tests:
- POST /api/v1/vehicles/decode-vin - VIN decoding
- GET /api/v1/vehicles/makes - Get vehicle makes
- GET /api/v1/vehicles/models/{make_id} - Get vehicle models
- GET /api/v1/vehicles/years - Get available years
- GET /api/v1/vehicles/{make}/{model}/{year}/recalls - Get vehicle recalls
- GET /api/v1/vehicles/{make}/{model}/{year}/complaints - Get vehicle complaints
"""

import pytest
from httpx import AsyncClient
from unittest.mock import patch, AsyncMock


class TestVINDecode:
    """Tests for POST /api/v1/vehicles/decode-vin endpoint."""

    @pytest.mark.asyncio
    async def test_decode_vin_success(
        self, async_client: AsyncClient, mock_nhtsa_service, valid_vins: list[str]
    ):
        """Test successful VIN decode returns 200."""
        with patch(
            "app.api.v1.endpoints.vehicles.get_nhtsa_service",
            return_value=mock_nhtsa_service,
        ):
            response = await async_client.post(
                "/api/v1/vehicles/decode-vin",
                json={"vin": valid_vins[0]},
            )

            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_decode_vin_returns_vehicle_info(
        self, async_client: AsyncClient, mock_nhtsa_service
    ):
        """Test that VIN decode returns vehicle information."""
        with patch(
            "app.api.v1.endpoints.vehicles.get_nhtsa_service",
            return_value=mock_nhtsa_service,
        ):
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
        self, async_client: AsyncClient, mock_nhtsa_service
    ):
        """Test that VIN is normalized to uppercase."""
        with patch(
            "app.api.v1.endpoints.vehicles.get_nhtsa_service",
            return_value=mock_nhtsa_service,
        ):
            response = await async_client.post(
                "/api/v1/vehicles/decode-vin",
                json={"vin": "wvwzzz3czwe123456"},  # lowercase
            )

            assert response.status_code == 200
            data = response.json()
            assert data["vin"] == "WVWZZZ3CZWE123456"

    @pytest.mark.asyncio
    async def test_decode_vin_too_short_returns_400(self, async_client: AsyncClient):
        """Test that VIN under 17 characters returns 400."""
        response = await async_client.post(
            "/api/v1/vehicles/decode-vin",
            json={"vin": "WVWZZZ3CZWE"},  # 11 chars
        )

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_decode_vin_too_long_returns_400(self, async_client: AsyncClient):
        """Test that VIN over 17 characters returns 400."""
        response = await async_client.post(
            "/api/v1/vehicles/decode-vin",
            json={"vin": "WVWZZZ3CZWE123456789"},  # 20 chars
        )

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_decode_vin_with_invalid_char_i_returns_400(self, async_client: AsyncClient):
        """Test that VIN containing 'I' returns 400."""
        response = await async_client.post(
            "/api/v1/vehicles/decode-vin",
            json={"vin": "WVWZZZ3CZWE12345I"},  # Contains I
        )

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_decode_vin_with_invalid_char_o_returns_400(self, async_client: AsyncClient):
        """Test that VIN containing 'O' returns 400."""
        response = await async_client.post(
            "/api/v1/vehicles/decode-vin",
            json={"vin": "WVWZZZ3CZWE12345O"},  # Contains O
        )

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_decode_vin_with_invalid_char_q_returns_400(self, async_client: AsyncClient):
        """Test that VIN containing 'Q' returns 400."""
        response = await async_client.post(
            "/api/v1/vehicles/decode-vin",
            json={"vin": "WVWZZZ3CZWE12345Q"},  # Contains Q
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
    """Tests for GET /api/v1/vehicles/makes endpoint."""

    @pytest.mark.asyncio
    async def test_get_makes_returns_200(self, async_client: AsyncClient):
        """Test getting vehicle makes returns 200."""
        response = await async_client.get("/api/v1/vehicles/makes")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_get_makes_returns_make_objects(self, async_client: AsyncClient):
        """Test that makes have correct structure."""
        response = await async_client.get("/api/v1/vehicles/makes")

        assert response.status_code == 200
        data = response.json()

        assert len(data) > 0
        make = data[0]
        assert "id" in make
        assert "name" in make
        assert "country" in make

    @pytest.mark.asyncio
    async def test_get_makes_includes_common_brands(self, async_client: AsyncClient):
        """Test that common car brands are included."""
        response = await async_client.get("/api/v1/vehicles/makes")

        assert response.status_code == 200
        data = response.json()

        make_names = [m["name"].lower() for m in data]
        # Check for some common brands
        assert "volkswagen" in make_names
        assert "toyota" in make_names
        assert "bmw" in make_names

    @pytest.mark.asyncio
    async def test_get_makes_with_search_filter(self, async_client: AsyncClient):
        """Test filtering makes by search term."""
        response = await async_client.get(
            "/api/v1/vehicles/makes",
            params={"search": "volk"},
        )

        assert response.status_code == 200
        data = response.json()

        # All results should contain "volk"
        for make in data:
            assert "volk" in make["name"].lower()

    @pytest.mark.asyncio
    async def test_get_makes_search_case_insensitive(self, async_client: AsyncClient):
        """Test that make search is case insensitive."""
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

        assert len(data_lower) == len(data_upper)

    @pytest.mark.asyncio
    async def test_get_makes_search_no_results(self, async_client: AsyncClient):
        """Test that search with no matches returns empty list."""
        response = await async_client.get(
            "/api/v1/vehicles/makes",
            params={"search": "zzzznonexistent"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data == []


class TestVehicleModels:
    """Tests for GET /api/v1/vehicles/models/{make_id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_models_returns_200(self, async_client: AsyncClient):
        """Test getting vehicle models returns 200."""
        response = await async_client.get("/api/v1/vehicles/models/volkswagen")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_get_models_returns_model_objects(self, async_client: AsyncClient):
        """Test that models have correct structure."""
        response = await async_client.get("/api/v1/vehicles/models/volkswagen")

        assert response.status_code == 200
        data = response.json()

        if data:
            model = data[0]
            assert "id" in model
            assert "name" in model
            assert "make_id" in model
            assert "year_start" in model

    @pytest.mark.asyncio
    async def test_get_models_includes_common_models(self, async_client: AsyncClient):
        """Test that common VW models are included."""
        response = await async_client.get("/api/v1/vehicles/models/volkswagen")

        assert response.status_code == 200
        data = response.json()

        model_names = [m["name"].lower() for m in data]
        assert "golf" in model_names or len(data) == 0

    @pytest.mark.asyncio
    async def test_get_models_with_year_filter(self, async_client: AsyncClient):
        """Test filtering models by year."""
        response = await async_client.get(
            "/api/v1/vehicles/models/volkswagen",
            params={"year": 2020},
        )

        assert response.status_code == 200
        data = response.json()

        # All models should be available in 2020
        for model in data:
            assert model["year_start"] <= 2020
            if model.get("year_end"):
                assert model["year_end"] >= 2020

    @pytest.mark.asyncio
    async def test_get_models_unknown_make_returns_empty(self, async_client: AsyncClient):
        """Test that unknown make returns empty list."""
        response = await async_client.get("/api/v1/vehicles/models/nonexistent_make")

        assert response.status_code == 200
        data = response.json()
        assert data == []


class TestVehicleYears:
    """Tests for GET /api/v1/vehicles/years endpoint."""

    @pytest.mark.asyncio
    async def test_get_years_returns_200(self, async_client: AsyncClient):
        """Test getting years returns 200."""
        response = await async_client.get("/api/v1/vehicles/years")

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_get_years_returns_years_list(self, async_client: AsyncClient):
        """Test that years response includes years list."""
        response = await async_client.get("/api/v1/vehicles/years")

        assert response.status_code == 200
        data = response.json()

        assert "years" in data
        assert isinstance(data["years"], list)

    @pytest.mark.asyncio
    async def test_get_years_includes_current_year(self, async_client: AsyncClient):
        """Test that current year is included."""
        from datetime import datetime

        response = await async_client.get("/api/v1/vehicles/years")

        assert response.status_code == 200
        data = response.json()

        current_year = datetime.now().year
        assert current_year in data["years"]

    @pytest.mark.asyncio
    async def test_get_years_includes_next_year(self, async_client: AsyncClient):
        """Test that next model year is included."""
        from datetime import datetime

        response = await async_client.get("/api/v1/vehicles/years")

        assert response.status_code == 200
        data = response.json()

        next_year = datetime.now().year + 1
        assert next_year in data["years"]

    @pytest.mark.asyncio
    async def test_get_years_sorted_descending(self, async_client: AsyncClient):
        """Test that years are sorted in descending order."""
        response = await async_client.get("/api/v1/vehicles/years")

        assert response.status_code == 200
        data = response.json()

        years = data["years"]
        assert years == sorted(years, reverse=True)

    @pytest.mark.asyncio
    async def test_get_years_starts_from_1980(self, async_client: AsyncClient):
        """Test that years go back to 1980."""
        response = await async_client.get("/api/v1/vehicles/years")

        assert response.status_code == 200
        data = response.json()

        assert 1980 in data["years"]


class TestVehicleRecalls:
    """Tests for GET /api/v1/vehicles/{make}/{model}/{year}/recalls endpoint."""

    @pytest.mark.asyncio
    async def test_get_recalls_returns_200(self, async_client: AsyncClient, mock_nhtsa_service):
        """Test getting recalls returns 200."""
        with patch(
            "app.api.v1.endpoints.vehicles.get_nhtsa_service",
            return_value=mock_nhtsa_service,
        ):
            response = await async_client.get("/api/v1/vehicles/Volkswagen/Golf/2018/recalls")

            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_get_recalls_returns_recall_objects(
        self, async_client: AsyncClient, mock_nhtsa_service
    ):
        """Test that recalls have correct structure."""
        with patch(
            "app.api.v1.endpoints.vehicles.get_nhtsa_service",
            return_value=mock_nhtsa_service,
        ):
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
    async def test_get_complaints_returns_200(self, async_client: AsyncClient, mock_nhtsa_service):
        """Test getting complaints returns 200."""
        with patch(
            "app.api.v1.endpoints.vehicles.get_nhtsa_service",
            return_value=mock_nhtsa_service,
        ):
            response = await async_client.get("/api/v1/vehicles/Volkswagen/Golf/2018/complaints")

            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_get_complaints_returns_complaint_objects(
        self, async_client: AsyncClient, mock_nhtsa_service
    ):
        """Test that complaints have correct structure."""
        with patch(
            "app.api.v1.endpoints.vehicles.get_nhtsa_service",
            return_value=mock_nhtsa_service,
        ):
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
    async def test_nhtsa_api_error_returns_502(self, async_client: AsyncClient):
        """Test that NHTSA API errors return 502."""
        from app.services.nhtsa_service import NHTSAError

        mock_service = AsyncMock()
        mock_service.decode_vin.side_effect = NHTSAError("Service unavailable")

        with patch(
            "app.api.v1.endpoints.vehicles.get_nhtsa_service",
            return_value=mock_service,
        ):
            response = await async_client.post(
                "/api/v1/vehicles/decode-vin",
                json={"vin": "WVWZZZ3CZWE123456"},
            )

            assert response.status_code == 502

    @pytest.mark.asyncio
    async def test_recalls_nhtsa_error_returns_502(self, async_client: AsyncClient):
        """Test that NHTSA API errors for recalls return 502."""
        from app.services.nhtsa_service import NHTSAError

        mock_service = AsyncMock()
        mock_service.get_recalls.side_effect = NHTSAError("Service unavailable")

        with patch(
            "app.api.v1.endpoints.vehicles.get_nhtsa_service",
            return_value=mock_service,
        ):
            response = await async_client.get("/api/v1/vehicles/Volkswagen/Golf/2018/recalls")

            assert response.status_code == 502

    @pytest.mark.asyncio
    async def test_complaints_nhtsa_error_returns_502(self, async_client: AsyncClient):
        """Test that NHTSA API errors for complaints return 502."""
        from app.services.nhtsa_service import NHTSAError

        mock_service = AsyncMock()
        mock_service.get_complaints.side_effect = NHTSAError("Service unavailable")

        with patch(
            "app.api.v1.endpoints.vehicles.get_nhtsa_service",
            return_value=mock_service,
        ):
            response = await async_client.get("/api/v1/vehicles/Volkswagen/Golf/2018/complaints")

            assert response.status_code == 502
