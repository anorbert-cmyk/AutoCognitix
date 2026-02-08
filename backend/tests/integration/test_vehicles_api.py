"""
Integration tests for the vehicle API endpoints.

Tests VIN decoding, vehicle makes/models, and NHTSA data retrieval.
"""

import pytest
from unittest.mock import AsyncMock, patch
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(backend_path))


class TestVINDecodeEndpoint:
    """Test POST /api/v1/vehicles/decode-vin endpoint."""

    @pytest.mark.asyncio
    async def test_decode_valid_vin_returns_200(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
    ):
        """Test that decoding valid VIN returns 200 OK."""
        with patch(
            "app.api.v1.endpoints.vehicles.get_nhtsa_service", return_value=mock_nhtsa_service
        ):
            response = await async_client.post(
                "/api/v1/vehicles/decode-vin",
                json={"vin": "WVWZZZ3CZWE123456"},
            )

            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_decode_valid_vin_returns_vehicle_info(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
    ):
        """Test that decoding valid VIN returns vehicle information."""
        with patch(
            "app.api.v1.endpoints.vehicles.get_nhtsa_service", return_value=mock_nhtsa_service
        ):
            response = await async_client.post(
                "/api/v1/vehicles/decode-vin",
                json={"vin": "WVWZZZ3CZWE123456"},
            )

            assert response.status_code == 200
            data = response.json()

            assert data["vin"] == "WVWZZZ3CZWE123456"
            assert data["make"] == "Volkswagen"
            assert data["model"] == "Golf"
            assert data["year"] == 2018

    @pytest.mark.asyncio
    async def test_decode_vin_returns_full_details(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
    ):
        """Test that decoded VIN includes full vehicle details."""
        with patch(
            "app.api.v1.endpoints.vehicles.get_nhtsa_service", return_value=mock_nhtsa_service
        ):
            response = await async_client.post(
                "/api/v1/vehicles/decode-vin",
                json={"vin": "WVWZZZ3CZWE123456"},
            )

            assert response.status_code == 200
            data = response.json()

            # Check optional fields are present
            assert "trim" in data
            assert "engine" in data
            assert "transmission" in data
            assert "drive_type" in data
            assert "body_type" in data
            assert "fuel_type" in data

    @pytest.mark.asyncio
    async def test_decode_short_vin_returns_400(self, async_client, seeded_db):
        """Test that short VIN returns 400 Bad Request."""
        response = await async_client.post(
            "/api/v1/vehicles/decode-vin",
            json={"vin": "WVWZZZ3CZWE"},  # Too short
        )

        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_decode_long_vin_returns_400(self, async_client, seeded_db):
        """Test that long VIN returns 400 Bad Request."""
        response = await async_client.post(
            "/api/v1/vehicles/decode-vin",
            json={"vin": "WVWZZZ3CZWE123456789"},  # Too long
        )

        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_decode_vin_with_invalid_characters(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
    ):
        """Test that VIN with invalid characters (I, O, Q) returns 400."""
        with patch(
            "app.api.v1.endpoints.vehicles.get_nhtsa_service", return_value=mock_nhtsa_service
        ):
            # VINs cannot contain I, O, or Q
            invalid_vins = [
                "WVWZZZ3CZWE12345I",  # Contains I
                "WVWZZZ3CZWE12345O",  # Contains O
                "WVWZZZ3CZWE12345Q",  # Contains Q
            ]

            for vin in invalid_vins:
                response = await async_client.post(
                    "/api/v1/vehicles/decode-vin",
                    json={"vin": vin},
                )
                assert response.status_code == 400, f"Expected 400 for VIN {vin}"

    @pytest.mark.asyncio
    async def test_decode_vin_normalizes_lowercase(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
    ):
        """Test that lowercase VIN is normalized to uppercase."""
        with patch(
            "app.api.v1.endpoints.vehicles.get_nhtsa_service", return_value=mock_nhtsa_service
        ):
            response = await async_client.post(
                "/api/v1/vehicles/decode-vin",
                json={"vin": "wvwzzz3czwe123456"},  # lowercase
            )

            assert response.status_code == 200
            data = response.json()
            # Should be normalized to uppercase
            assert data["vin"] == "WVWZZZ3CZWE123456"

    @pytest.mark.asyncio
    async def test_decode_invalid_vin_returns_400(
        self,
        async_client,
        seeded_db,
    ):
        """Test that NHTSA-invalid VIN returns 400."""
        from app.services.nhtsa_service import VINDecodeResult

        invalid_result = VINDecodeResult(
            vin="12345678901234567",
            error_code="5",
            error_text="Invalid VIN: Cannot decode",
            raw_data={},
        )

        mock_nhtsa = AsyncMock()
        mock_nhtsa.decode_vin.return_value = invalid_result

        with patch("app.api.v1.endpoints.vehicles.get_nhtsa_service", return_value=mock_nhtsa):
            response = await async_client.post(
                "/api/v1/vehicles/decode-vin",
                json={"vin": "12345678901234567"},
            )

            assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_decode_vin_handles_nhtsa_error(self, async_client, seeded_db):
        """Test that NHTSA API error is handled gracefully."""
        from app.services.nhtsa_service import NHTSAError

        mock_nhtsa = AsyncMock()
        mock_nhtsa.decode_vin.side_effect = NHTSAError("Service unavailable", status_code=503)

        with patch("app.api.v1.endpoints.vehicles.get_nhtsa_service", return_value=mock_nhtsa):
            response = await async_client.post(
                "/api/v1/vehicles/decode-vin",
                json={"vin": "WVWZZZ3CZWE123456"},
            )

            assert response.status_code == 502  # Bad Gateway


class TestVehicleMakesEndpoint:
    """Test GET /api/v1/vehicles/makes endpoint."""

    @pytest.mark.asyncio
    async def test_get_makes_returns_200(self, async_client, seeded_db, mock_vehicle_service):
        """Test that getting makes returns 200 OK."""
        with patch(
            "app.api.v1.endpoints.vehicles.get_vehicle_service", return_value=mock_vehicle_service
        ):
            response = await async_client.get("/api/v1/vehicles/makes")

            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_get_makes_returns_paginated_response(
        self, async_client, seeded_db, mock_vehicle_service
    ):
        """Test that makes endpoint returns a paginated response."""
        with patch(
            "app.api.v1.endpoints.vehicles.get_vehicle_service", return_value=mock_vehicle_service
        ):
            response = await async_client.get("/api/v1/vehicles/makes")

            assert response.status_code == 200
            data = response.json()
            assert "items" in data
            assert "total" in data
            assert "limit" in data
            assert "offset" in data
            assert "has_more" in data
            assert isinstance(data["items"], list)

    @pytest.mark.asyncio
    async def test_get_makes_includes_common_manufacturers(
        self, async_client, seeded_db, mock_vehicle_service
    ):
        """Test that makes list includes common manufacturers."""
        with patch(
            "app.api.v1.endpoints.vehicles.get_vehicle_service", return_value=mock_vehicle_service
        ):
            response = await async_client.get("/api/v1/vehicles/makes")

            assert response.status_code == 200
            data = response.json()

            make_names = [make["name"] for make in data["items"]]
            # Check for common European makes (popular in Hungary)
            assert "Volkswagen" in make_names
            assert "BMW" in make_names
            assert "Toyota" in make_names

    @pytest.mark.asyncio
    async def test_get_makes_have_required_fields(
        self, async_client, seeded_db, mock_vehicle_service
    ):
        """Test that makes have required fields."""
        with patch(
            "app.api.v1.endpoints.vehicles.get_vehicle_service", return_value=mock_vehicle_service
        ):
            response = await async_client.get("/api/v1/vehicles/makes")

            assert response.status_code == 200
            data = response.json()

            for make in data["items"]:
                assert "id" in make
                assert "name" in make

    @pytest.mark.asyncio
    async def test_get_makes_with_search_filter(
        self, async_client, seeded_db, mock_vehicle_service
    ):
        """Test that makes can be filtered by search term."""
        with patch(
            "app.api.v1.endpoints.vehicles.get_vehicle_service", return_value=mock_vehicle_service
        ):
            response = await async_client.get(
                "/api/v1/vehicles/makes",
                params={"search": "volks"},
            )

            assert response.status_code == 200
            data = response.json()

            # Should find Volkswagen in items
            assert len(data["items"]) >= 1
            names = [make["name"].lower() for make in data["items"]]
            assert any("volks" in name for name in names)


class TestVehicleModelsEndpoint:
    """Test GET /api/v1/vehicles/models endpoint."""

    @pytest.mark.asyncio
    async def test_get_models_returns_200(self, async_client, seeded_db, mock_vehicle_service):
        """Test that getting models returns 200 OK."""
        with patch(
            "app.api.v1.endpoints.vehicles.get_vehicle_service", return_value=mock_vehicle_service
        ):
            response = await async_client.get(
                "/api/v1/vehicles/models",
                params={"make": "Volkswagen"},
            )

            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_get_models_returns_paginated_response(
        self, async_client, seeded_db, mock_vehicle_service
    ):
        """Test that models endpoint returns a paginated response."""
        with patch(
            "app.api.v1.endpoints.vehicles.get_vehicle_service", return_value=mock_vehicle_service
        ):
            response = await async_client.get(
                "/api/v1/vehicles/models",
                params={"make": "Volkswagen"},
            )

            assert response.status_code == 200
            data = response.json()
            assert "items" in data
            assert "total" in data
            assert isinstance(data["items"], list)

    @pytest.mark.asyncio
    async def test_get_models_for_volkswagen(self, async_client, seeded_db, mock_vehicle_service):
        """Test that Volkswagen models include common models."""
        with patch(
            "app.api.v1.endpoints.vehicles.get_vehicle_service", return_value=mock_vehicle_service
        ):
            response = await async_client.get(
                "/api/v1/vehicles/models",
                params={"make": "Volkswagen"},
            )

            assert response.status_code == 200
            data = response.json()

            if data["items"]:
                model_names = [model["name"] for model in data["items"]]
                # Should include common VW models
                assert "Golf" in model_names or "Passat" in model_names

    @pytest.mark.asyncio
    async def test_get_models_have_required_fields(
        self, async_client, seeded_db, mock_vehicle_service
    ):
        """Test that models have required fields."""
        with patch(
            "app.api.v1.endpoints.vehicles.get_vehicle_service", return_value=mock_vehicle_service
        ):
            response = await async_client.get(
                "/api/v1/vehicles/models",
                params={"make": "Volkswagen"},
            )

            assert response.status_code == 200
            data = response.json()

            for model in data["items"]:
                assert "id" in model
                assert "name" in model
                assert "make_id" in model

    @pytest.mark.asyncio
    async def test_get_models_unknown_make_returns_404(self, async_client, seeded_db):
        """Test that unknown make returns 404."""
        # Mock vehicle service that returns empty results
        mock_service = AsyncMock()
        mock_service.get_models_for_make.return_value = ([], 0)

        with patch("app.api.v1.endpoints.vehicles.get_vehicle_service", return_value=mock_service):
            response = await async_client.get(
                "/api/v1/vehicles/models",
                params={"make": "UnknownMake"},
            )

            assert response.status_code == 404


class TestVehicleYearsEndpoint:
    """Test GET /api/v1/vehicles/years endpoint."""

    @pytest.mark.asyncio
    async def test_get_years_returns_200(self, async_client, seeded_db, mock_vehicle_service):
        """Test that getting years returns 200 OK."""
        with patch(
            "app.api.v1.endpoints.vehicles.get_vehicle_service", return_value=mock_vehicle_service
        ):
            response = await async_client.get(
                "/api/v1/vehicles/years",
                params={"make": "Volkswagen", "model": "Golf"},
            )

            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_get_years_returns_list(self, async_client, seeded_db, mock_vehicle_service):
        """Test that years endpoint returns year list."""
        with patch(
            "app.api.v1.endpoints.vehicles.get_vehicle_service", return_value=mock_vehicle_service
        ):
            response = await async_client.get(
                "/api/v1/vehicles/years",
                params={"make": "Volkswagen", "model": "Golf"},
            )

            assert response.status_code == 200
            data = response.json()

            assert "years" in data
            assert "make" in data
            assert "model" in data
            assert isinstance(data["years"], list)
            assert len(data["years"]) > 0

    @pytest.mark.asyncio
    async def test_get_years_sorted_descending(self, async_client, seeded_db, mock_vehicle_service):
        """Test that years are sorted in descending order."""
        with patch(
            "app.api.v1.endpoints.vehicles.get_vehicle_service", return_value=mock_vehicle_service
        ):
            response = await async_client.get(
                "/api/v1/vehicles/years",
                params={"make": "Volkswagen", "model": "Golf"},
            )

            assert response.status_code == 200
            data = response.json()

            years = data["years"]
            # First year should be most recent
            if len(years) > 1:
                assert years[0] > years[-1]

    @pytest.mark.asyncio
    async def test_get_years_requires_make_param(self, async_client, seeded_db):
        """Test that make parameter is required."""
        response = await async_client.get(
            "/api/v1/vehicles/years",
            params={"model": "Golf"},  # Missing make
        )

        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_get_years_requires_model_param(self, async_client, seeded_db):
        """Test that model parameter is required."""
        response = await async_client.get(
            "/api/v1/vehicles/years",
            params={"make": "Volkswagen"},  # Missing model
        )

        assert response.status_code == 422  # Validation error


class TestVehicleRecallsEndpoint:
    """Test GET /api/v1/vehicles/{make}/{model}/{year}/recalls endpoint."""

    @pytest.mark.asyncio
    async def test_get_recalls_returns_200(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
    ):
        """Test that getting recalls returns 200 OK."""
        with patch(
            "app.api.v1.endpoints.vehicles.get_nhtsa_service", return_value=mock_nhtsa_service
        ):
            response = await async_client.get("/api/v1/vehicles/Volkswagen/Golf/2018/recalls")

            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_get_recalls_returns_list(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
    ):
        """Test that recalls endpoint returns a list."""
        with patch(
            "app.api.v1.endpoints.vehicles.get_nhtsa_service", return_value=mock_nhtsa_service
        ):
            response = await async_client.get("/api/v1/vehicles/Volkswagen/Golf/2018/recalls")

            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_get_recalls_have_required_fields(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
    ):
        """Test that recalls have required fields."""
        with patch(
            "app.api.v1.endpoints.vehicles.get_nhtsa_service", return_value=mock_nhtsa_service
        ):
            response = await async_client.get("/api/v1/vehicles/Volkswagen/Golf/2018/recalls")

            assert response.status_code == 200
            data = response.json()

            for recall in data:
                assert "campaign_number" in recall
                assert "component" in recall
                assert "summary" in recall

    @pytest.mark.asyncio
    async def test_get_recalls_handles_nhtsa_error(self, async_client, seeded_db):
        """Test that NHTSA API error is handled gracefully."""
        from app.services.nhtsa_service import NHTSAError

        mock_nhtsa = AsyncMock()
        mock_nhtsa.get_recalls.side_effect = NHTSAError("Service unavailable", status_code=503)

        with patch("app.api.v1.endpoints.vehicles.get_nhtsa_service", return_value=mock_nhtsa):
            response = await async_client.get("/api/v1/vehicles/Volkswagen/Golf/2018/recalls")

            assert response.status_code == 502

    @pytest.mark.asyncio
    async def test_get_recalls_validates_year_range(self, async_client, seeded_db):
        """Test that year parameter is validated."""
        response = await async_client.get(
            "/api/v1/vehicles/Volkswagen/Golf/1800/recalls"  # Invalid year
        )

        assert response.status_code == 422


class TestVehicleComplaintsEndpoint:
    """Test GET /api/v1/vehicles/{make}/{model}/{year}/complaints endpoint."""

    @pytest.mark.asyncio
    async def test_get_complaints_returns_200(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
    ):
        """Test that getting complaints returns 200 OK."""
        with patch(
            "app.api.v1.endpoints.vehicles.get_nhtsa_service", return_value=mock_nhtsa_service
        ):
            response = await async_client.get("/api/v1/vehicles/Volkswagen/Golf/2018/complaints")

            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_get_complaints_returns_list(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
    ):
        """Test that complaints endpoint returns a list."""
        with patch(
            "app.api.v1.endpoints.vehicles.get_nhtsa_service", return_value=mock_nhtsa_service
        ):
            response = await async_client.get("/api/v1/vehicles/Volkswagen/Golf/2018/complaints")

            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_get_complaints_have_required_fields(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
    ):
        """Test that complaints have required fields."""
        with patch(
            "app.api.v1.endpoints.vehicles.get_nhtsa_service", return_value=mock_nhtsa_service
        ):
            response = await async_client.get("/api/v1/vehicles/Volkswagen/Golf/2018/complaints")

            assert response.status_code == 200
            data = response.json()

            for complaint in data:
                assert "make" in complaint
                assert "model" in complaint
                assert "model_year" in complaint

    @pytest.mark.asyncio
    async def test_get_complaints_include_safety_info(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
    ):
        """Test that complaints include safety information."""
        with patch(
            "app.api.v1.endpoints.vehicles.get_nhtsa_service", return_value=mock_nhtsa_service
        ):
            response = await async_client.get("/api/v1/vehicles/Volkswagen/Golf/2018/complaints")

            assert response.status_code == 200
            data = response.json()

            for complaint in data:
                assert "crash" in complaint
                assert "fire" in complaint
                assert "injuries" in complaint
                assert "deaths" in complaint

    @pytest.mark.asyncio
    async def test_get_complaints_handles_nhtsa_error(self, async_client, seeded_db):
        """Test that NHTSA API error is handled gracefully."""
        from app.services.nhtsa_service import NHTSAError

        mock_nhtsa = AsyncMock()
        mock_nhtsa.get_complaints.side_effect = NHTSAError("Service unavailable", status_code=503)

        with patch("app.api.v1.endpoints.vehicles.get_nhtsa_service", return_value=mock_nhtsa):
            response = await async_client.get("/api/v1/vehicles/Volkswagen/Golf/2018/complaints")

            assert response.status_code == 502


class TestVINValidation:
    """Test VIN validation logic."""

    @pytest.mark.asyncio
    async def test_vin_length_exactly_17(self, async_client, seeded_db, valid_vins):
        """Test that all valid VINs are exactly 17 characters."""
        for vin in valid_vins:
            assert len(vin) == 17

    @pytest.mark.asyncio
    async def test_vin_no_invalid_characters(self, async_client, seeded_db, valid_vins):
        """Test that valid VINs don't contain I, O, or Q."""
        invalid_chars = set("IOQ")
        for vin in valid_vins:
            assert not any(char in invalid_chars for char in vin)

    @pytest.mark.asyncio
    async def test_invalid_vins_rejected(
        self,
        async_client,
        seeded_db,
        invalid_vins,
    ):
        """Test that invalid VINs are rejected."""
        for vin in invalid_vins:
            response = await async_client.post(
                "/api/v1/vehicles/decode-vin",
                json={"vin": vin},
            )
            # Should return validation error (422) or bad request (400)
            assert response.status_code in [400, 422], f"VIN {vin} should be rejected"
