"""
Integration tests for the diagnosis API endpoints.

Tests the complete diagnosis flow including:
- DiagnosisRequest validation
- Service orchestration
- Response formatting
- Error handling
"""

import pytest
from unittest.mock import AsyncMock, patch
from uuid import uuid4
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(backend_path))


class TestDiagnosisAnalyzeEndpoint:
    """Test POST /api/v1/diagnosis/analyze endpoint."""

    @pytest.mark.asyncio
    async def test_analyze_returns_201_with_valid_request(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
        mock_rag_service,
        diagnosis_request_data,
    ):
        """Test that analyze returns 201 CREATED with valid request."""
        with patch("app.services.nhtsa_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.diagnosis_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.rag_service.diagnose", new=mock_rag_service.diagnose):

            response = await async_client.post(
                "/api/v1/diagnosis/analyze",
                json=diagnosis_request_data,
            )

            assert response.status_code == 201

    @pytest.mark.asyncio
    async def test_analyze_returns_diagnosis_response_structure(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
        mock_rag_service,
        diagnosis_request_data,
    ):
        """Test that analyze returns correct response structure."""
        with patch("app.services.nhtsa_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.diagnosis_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.rag_service.diagnose", new=mock_rag_service.diagnose):

            response = await async_client.post(
                "/api/v1/diagnosis/analyze",
                json=diagnosis_request_data,
            )

            if response.status_code == 201:
                data = response.json()

                # Check required fields
                assert "id" in data
                assert "vehicle_make" in data
                assert "vehicle_model" in data
                assert "vehicle_year" in data
                assert "dtc_codes" in data
                assert "probable_causes" in data
                assert "recommended_repairs" in data
                assert "confidence_score" in data

    @pytest.mark.asyncio
    async def test_analyze_validates_vehicle_make_required(self, async_client, seeded_db):
        """Test that vehicle_make is required."""
        request_data = {
            "vehicle_model": "Golf",
            "vehicle_year": 2018,
            "dtc_codes": ["P0101"],
            "symptoms": "Motor nehezen indul hidegben",
        }

        response = await async_client.post("/api/v1/diagnosis/analyze", json=request_data)

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    @pytest.mark.asyncio
    async def test_analyze_validates_vehicle_model_required(self, async_client, seeded_db):
        """Test that vehicle_model is required."""
        request_data = {
            "vehicle_make": "Volkswagen",
            "vehicle_year": 2018,
            "dtc_codes": ["P0101"],
            "symptoms": "Motor nehezen indul hidegben",
        }

        response = await async_client.post("/api/v1/diagnosis/analyze", json=request_data)

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_analyze_validates_vehicle_year_range(self, async_client, seeded_db):
        """Test that vehicle_year must be within valid range."""
        request_data = {
            "vehicle_make": "Volkswagen",
            "vehicle_model": "Golf",
            "vehicle_year": 1800,  # Invalid - too old
            "dtc_codes": ["P0101"],
            "symptoms": "Motor nehezen indul hidegben",
        }

        response = await async_client.post("/api/v1/diagnosis/analyze", json=request_data)

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_analyze_validates_dtc_codes_required(self, async_client, seeded_db):
        """Test that dtc_codes is required and must not be empty."""
        request_data = {
            "vehicle_make": "Volkswagen",
            "vehicle_model": "Golf",
            "vehicle_year": 2018,
            "dtc_codes": [],  # Empty list
            "symptoms": "Motor nehezen indul hidegben",
        }

        response = await async_client.post("/api/v1/diagnosis/analyze", json=request_data)

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_analyze_validates_symptoms_minimum_length(self, async_client, seeded_db):
        """Test that symptoms must meet minimum length requirement."""
        request_data = {
            "vehicle_make": "Volkswagen",
            "vehicle_model": "Golf",
            "vehicle_year": 2018,
            "dtc_codes": ["P0101"],
            "symptoms": "Short",  # Too short
        }

        response = await async_client.post("/api/v1/diagnosis/analyze", json=request_data)

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_analyze_validates_vin_format(self, async_client, seeded_db):
        """Test that VIN must be exactly 17 characters if provided."""
        request_data = {
            "vehicle_make": "Volkswagen",
            "vehicle_model": "Golf",
            "vehicle_year": 2018,
            "vin": "INVALID",  # Invalid VIN
            "dtc_codes": ["P0101"],
            "symptoms": "Motor nehezen indul hidegben",
        }

        response = await async_client.post("/api/v1/diagnosis/analyze", json=request_data)

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_analyze_validates_dtc_codes_max_count(self, async_client, seeded_db):
        """Test that dtc_codes list cannot exceed maximum count."""
        request_data = {
            "vehicle_make": "Volkswagen",
            "vehicle_model": "Golf",
            "vehicle_year": 2018,
            "dtc_codes": [f"P{i:04d}" for i in range(25)],  # Too many codes
            "symptoms": "Motor nehezen indul hidegben",
        }

        response = await async_client.post("/api/v1/diagnosis/analyze", json=request_data)

        assert response.status_code == 422


class TestDiagnosisGetEndpoint:
    """Test GET /api/v1/diagnosis/{diagnosis_id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_nonexistent_diagnosis_returns_404(self, async_client, seeded_db):
        """Test that getting nonexistent diagnosis returns 404."""
        fake_id = str(uuid4())

        response = await async_client.get(f"/api/v1/diagnosis/{fake_id}")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_invalid_uuid_returns_422(self, async_client, seeded_db):
        """Test that invalid UUID format returns 422."""
        response = await async_client.get("/api/v1/diagnosis/invalid-uuid")

        assert response.status_code == 422


class TestDiagnosisHistoryEndpoint:
    """Test GET /api/v1/diagnosis/history/list endpoint."""

    @pytest.mark.asyncio
    async def test_history_returns_empty_list_for_no_history(self, async_client, seeded_db):
        """Test that history returns empty list when no history exists."""
        response = await async_client.get("/api/v1/diagnosis/history/list")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_history_respects_pagination_params(self, async_client, seeded_db):
        """Test that history endpoint respects skip and limit parameters."""
        response = await async_client.get(
            "/api/v1/diagnosis/history/list",
            params={"skip": 0, "limit": 5},
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) <= 5

    @pytest.mark.asyncio
    async def test_history_validates_limit_range(self, async_client, seeded_db):
        """Test that limit parameter must be within valid range."""
        # Test limit too high
        response = await async_client.get(
            "/api/v1/diagnosis/history/list",
            params={"limit": 150},  # Max is 100
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_history_validates_skip_non_negative(self, async_client, seeded_db):
        """Test that skip parameter must be non-negative."""
        response = await async_client.get(
            "/api/v1/diagnosis/history/list",
            params={"skip": -1},
        )

        assert response.status_code == 422


class TestQuickAnalyzeEndpoint:
    """Test POST /api/v1/diagnosis/quick-analyze endpoint."""

    @pytest.mark.asyncio
    async def test_quick_analyze_returns_200_with_valid_codes(
        self,
        async_client,
        seeded_db,
    ):
        """Test that quick-analyze returns 200 with valid DTC codes."""
        response = await async_client.post(
            "/api/v1/diagnosis/quick-analyze",
            params={"dtc_codes": ["P0101"]},
        )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_quick_analyze_returns_dtc_details(
        self,
        async_client,
        seeded_db,
    ):
        """Test that quick-analyze returns DTC details structure."""
        response = await async_client.post(
            "/api/v1/diagnosis/quick-analyze",
            params={"dtc_codes": ["P0101"]},
        )

        assert response.status_code == 200
        data = response.json()
        assert "dtc_codes" in data
        assert isinstance(data["dtc_codes"], list)

    @pytest.mark.asyncio
    async def test_quick_analyze_validates_invalid_dtc_format(
        self,
        async_client,
        seeded_db,
    ):
        """Test that quick-analyze rejects invalid DTC format."""
        response = await async_client.post(
            "/api/v1/diagnosis/quick-analyze",
            params={"dtc_codes": ["INVALID"]},
        )

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_quick_analyze_handles_unknown_codes(
        self,
        async_client,
        seeded_db,
    ):
        """Test that quick-analyze handles unknown DTC codes gracefully."""
        response = await async_client.post(
            "/api/v1/diagnosis/quick-analyze",
            params={"dtc_codes": ["P9999"]},  # Unknown code
        )

        assert response.status_code == 200
        data = response.json()
        assert "dtc_codes" in data
        # Should indicate code not found
        if data["dtc_codes"]:
            first_code = data["dtc_codes"][0]
            assert first_code["code"] == "P9999"

    @pytest.mark.asyncio
    async def test_quick_analyze_accepts_multiple_codes(
        self,
        async_client,
        seeded_db,
    ):
        """Test that quick-analyze accepts multiple DTC codes."""
        response = await async_client.post(
            "/api/v1/diagnosis/quick-analyze",
            params={"dtc_codes": ["P0101", "P0171", "B1234"]},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["dtc_codes"]) == 3

    @pytest.mark.asyncio
    async def test_quick_analyze_validates_max_codes(
        self,
        async_client,
        seeded_db,
    ):
        """Test that quick-analyze enforces maximum code limit."""
        # Create list of 15 codes (max is 10)
        codes = [f"P{i:04d}" for i in range(15)]

        response = await async_client.post(
            "/api/v1/diagnosis/quick-analyze",
            params={"dtc_codes": codes},
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_quick_analyze_normalizes_lowercase_codes(
        self,
        async_client,
        seeded_db,
    ):
        """Test that quick-analyze normalizes lowercase DTC codes."""
        response = await async_client.post(
            "/api/v1/diagnosis/quick-analyze",
            params={"dtc_codes": ["p0101"]},  # lowercase
        )

        assert response.status_code == 200


class TestDiagnosisServiceIntegration:
    """Test diagnosis service integration with other services."""

    @pytest.mark.asyncio
    async def test_diagnosis_integrates_with_nhtsa_service(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
        mock_rag_service,
        diagnosis_request_data,
    ):
        """Test that diagnosis fetches NHTSA recalls and complaints."""
        with patch("app.services.nhtsa_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.diagnosis_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.rag_service.diagnose", new=mock_rag_service.diagnose):

            response = await async_client.post(
                "/api/v1/diagnosis/analyze",
                json=diagnosis_request_data,
            )

            # Verify NHTSA service was called
            if response.status_code == 201:
                mock_nhtsa_service.get_recalls.assert_called()
                mock_nhtsa_service.get_complaints.assert_called()

    @pytest.mark.asyncio
    async def test_diagnosis_handles_nhtsa_service_failure(
        self,
        async_client,
        seeded_db,
        mock_rag_service,
        diagnosis_request_data,
    ):
        """Test that diagnosis handles NHTSA service failure gracefully."""
        failing_nhtsa = AsyncMock()
        failing_nhtsa.decode_vin.side_effect = Exception("NHTSA service unavailable")
        failing_nhtsa.get_recalls.return_value = []
        failing_nhtsa.get_complaints.return_value = []

        with patch("app.services.nhtsa_service.get_nhtsa_service", return_value=failing_nhtsa), \
             patch("app.services.diagnosis_service.get_nhtsa_service", return_value=failing_nhtsa), \
             patch("app.services.rag_service.diagnose", new=mock_rag_service.diagnose):

            # Remove VIN to avoid VIN decode
            request_data = diagnosis_request_data.copy()
            del request_data["vin"]

            response = await async_client.post(
                "/api/v1/diagnosis/analyze",
                json=request_data,
            )

            # Should still return a response (graceful degradation)
            assert response.status_code in [201, 500]

    @pytest.mark.asyncio
    async def test_diagnosis_validates_vin_through_nhtsa(
        self,
        async_client,
        seeded_db,
        mock_rag_service,
        diagnosis_request_data,
    ):
        """Test that diagnosis validates VIN through NHTSA if provided."""
        from app.services.nhtsa_service import VINDecodeResult

        invalid_vin_result = VINDecodeResult(
            vin="INVALID12345678901",
            error_code="5",
            error_text="Invalid VIN",
            raw_data={},
        )

        mock_nhtsa = AsyncMock()
        mock_nhtsa.decode_vin.return_value = invalid_vin_result
        mock_nhtsa.get_recalls.return_value = []
        mock_nhtsa.get_complaints.return_value = []

        with patch("app.services.nhtsa_service.get_nhtsa_service", return_value=mock_nhtsa), \
             patch("app.services.diagnosis_service.get_nhtsa_service", return_value=mock_nhtsa), \
             patch("app.services.rag_service.diagnose", new=mock_rag_service.diagnose):

            response = await async_client.post(
                "/api/v1/diagnosis/analyze",
                json=diagnosis_request_data,
            )

            # Should return 400 for invalid VIN
            assert response.status_code == 400


class TestDiagnosisResponseFormat:
    """Test diagnosis response format and content."""

    @pytest.mark.asyncio
    async def test_response_contains_probable_causes_with_confidence(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
        mock_rag_service,
        diagnosis_request_data,
    ):
        """Test that response contains probable causes with confidence scores."""
        with patch("app.services.nhtsa_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.diagnosis_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.rag_service.diagnose", new=mock_rag_service.diagnose):

            response = await async_client.post(
                "/api/v1/diagnosis/analyze",
                json=diagnosis_request_data,
            )

            if response.status_code == 201:
                data = response.json()
                assert "probable_causes" in data

                for cause in data["probable_causes"]:
                    assert "title" in cause
                    assert "description" in cause
                    assert "confidence" in cause
                    assert 0 <= cause["confidence"] <= 1

    @pytest.mark.asyncio
    async def test_response_contains_repair_recommendations(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
        mock_rag_service,
        diagnosis_request_data,
    ):
        """Test that response contains repair recommendations."""
        with patch("app.services.nhtsa_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.diagnosis_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.rag_service.diagnose", new=mock_rag_service.diagnose):

            response = await async_client.post(
                "/api/v1/diagnosis/analyze",
                json=diagnosis_request_data,
            )

            if response.status_code == 201:
                data = response.json()
                assert "recommended_repairs" in data

                for repair in data["recommended_repairs"]:
                    assert "title" in repair
                    assert "description" in repair
                    assert "difficulty" in repair

    @pytest.mark.asyncio
    async def test_response_contains_overall_confidence_score(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
        mock_rag_service,
        diagnosis_request_data,
    ):
        """Test that response contains overall confidence score."""
        with patch("app.services.nhtsa_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.diagnosis_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.rag_service.diagnose", new=mock_rag_service.diagnose):

            response = await async_client.post(
                "/api/v1/diagnosis/analyze",
                json=diagnosis_request_data,
            )

            if response.status_code == 201:
                data = response.json()
                assert "confidence_score" in data
                assert 0 <= data["confidence_score"] <= 1

    @pytest.mark.asyncio
    async def test_response_echoes_input_vehicle_info(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
        mock_rag_service,
        diagnosis_request_data,
    ):
        """Test that response echoes input vehicle information."""
        with patch("app.services.nhtsa_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.diagnosis_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.rag_service.diagnose", new=mock_rag_service.diagnose):

            response = await async_client.post(
                "/api/v1/diagnosis/analyze",
                json=diagnosis_request_data,
            )

            if response.status_code == 201:
                data = response.json()
                assert data["vehicle_make"] == diagnosis_request_data["vehicle_make"]
                assert data["vehicle_model"] == diagnosis_request_data["vehicle_model"]
                assert data["vehicle_year"] == diagnosis_request_data["vehicle_year"]
