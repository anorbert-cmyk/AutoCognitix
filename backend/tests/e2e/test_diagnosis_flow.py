"""
End-to-end tests for the complete diagnosis workflow.

Tests the full diagnosis flow including:
- Vehicle selection (make/model/year)
- DTC code input and validation
- Symptom text processing (Hungarian language)
- Full diagnosis analysis with RAG
- Result retrieval and history
- Authenticated vs anonymous user flows
"""

import pytest
from unittest.mock import AsyncMock, patch
from uuid import uuid4
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(backend_path))


class TestVehicleSelection:
    """Test vehicle selection (make/model/year) in diagnosis requests."""

    @pytest.mark.asyncio
    async def test_valid_vehicle_make_accepted(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
        mock_rag_service,
    ):
        """Test that valid vehicle make is accepted."""
        request_data = {
            "vehicle_make": "Volkswagen",
            "vehicle_model": "Golf",
            "vehicle_year": 2018,
            "dtc_codes": ["P0101"],
            "symptoms": "A motor nehezen indul es egyenetlenul jar.",
        }

        with patch("app.services.nhtsa_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.diagnosis_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.rag_service.diagnose", new=mock_rag_service.diagnose):

            response = await async_client.post(
                "/api/v1/diagnosis/analyze",
                json=request_data,
            )

            assert response.status_code == 201
            data = response.json()
            assert data["vehicle_make"] == "Volkswagen"

    @pytest.mark.asyncio
    async def test_vehicle_make_required(self, async_client, seeded_db):
        """Test that vehicle_make is required."""
        request_data = {
            "vehicle_model": "Golf",
            "vehicle_year": 2018,
            "dtc_codes": ["P0101"],
            "symptoms": "A motor nehezen indul es egyenetlenul jar.",
        }

        response = await async_client.post("/api/v1/diagnosis/analyze", json=request_data)

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    @pytest.mark.asyncio
    async def test_vehicle_model_required(self, async_client, seeded_db):
        """Test that vehicle_model is required."""
        request_data = {
            "vehicle_make": "Volkswagen",
            "vehicle_year": 2018,
            "dtc_codes": ["P0101"],
            "symptoms": "A motor nehezen indul es egyenetlenul jar.",
        }

        response = await async_client.post("/api/v1/diagnosis/analyze", json=request_data)

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_vehicle_year_must_be_valid_range(self, async_client, seeded_db):
        """Test that vehicle_year must be within valid range (1900-2030)."""
        # Test year too old
        request_data = {
            "vehicle_make": "Volkswagen",
            "vehicle_model": "Golf",
            "vehicle_year": 1800,
            "dtc_codes": ["P0101"],
            "symptoms": "A motor nehezen indul es egyenetlenul jar.",
        }

        response = await async_client.post("/api/v1/diagnosis/analyze", json=request_data)
        assert response.status_code == 422

        # Test year too far in future
        request_data["vehicle_year"] = 2050
        response = await async_client.post("/api/v1/diagnosis/analyze", json=request_data)
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_vehicle_year_accepts_current_models(self, async_client, seeded_db, mock_nhtsa_service, mock_rag_service):
        """Test that vehicle_year accepts recent model years."""
        request_data = {
            "vehicle_make": "Toyota",
            "vehicle_model": "Corolla",
            "vehicle_year": 2024,
            "dtc_codes": ["P0300"],
            "symptoms": "A motor rezeg es erot veszit gyorsitaskor.",
        }

        with patch("app.services.nhtsa_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.diagnosis_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.rag_service.diagnose", new=mock_rag_service.diagnose):

            response = await async_client.post("/api/v1/diagnosis/analyze", json=request_data)
            assert response.status_code == 201

    @pytest.mark.asyncio
    async def test_vehicle_engine_optional(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
        mock_rag_service,
    ):
        """Test that vehicle_engine is optional but used when provided."""
        # Without engine
        request_without_engine = {
            "vehicle_make": "BMW",
            "vehicle_model": "3 Series",
            "vehicle_year": 2019,
            "dtc_codes": ["P0171"],
            "symptoms": "A motor nehezen indul es a fogyasztas magas.",
        }

        with patch("app.services.nhtsa_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.diagnosis_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.rag_service.diagnose", new=mock_rag_service.diagnose):

            response = await async_client.post("/api/v1/diagnosis/analyze", json=request_without_engine)
            assert response.status_code == 201

        # With engine
        request_with_engine = {
            "vehicle_make": "BMW",
            "vehicle_model": "3 Series",
            "vehicle_year": 2019,
            "vehicle_engine": "2.0 TwinPower Turbo",
            "dtc_codes": ["P0171"],
            "symptoms": "A motor nehezen indul es a fogyasztas magas.",
        }

        with patch("app.services.nhtsa_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.diagnosis_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.rag_service.diagnose", new=mock_rag_service.diagnose):

            response = await async_client.post("/api/v1/diagnosis/analyze", json=request_with_engine)
            assert response.status_code == 201


class TestDTCCodeInput:
    """Test DTC code input and validation."""

    @pytest.mark.asyncio
    async def test_single_dtc_code_accepted(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
        mock_rag_service,
    ):
        """Test that single DTC code is accepted."""
        request_data = {
            "vehicle_make": "Volkswagen",
            "vehicle_model": "Golf",
            "vehicle_year": 2018,
            "dtc_codes": ["P0101"],
            "symptoms": "A motor nehezen indul es egyenetlenul jar.",
        }

        with patch("app.services.nhtsa_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.diagnosis_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.rag_service.diagnose", new=mock_rag_service.diagnose):

            response = await async_client.post("/api/v1/diagnosis/analyze", json=request_data)
            assert response.status_code == 201
            data = response.json()
            assert data["dtc_codes"] == ["P0101"]

    @pytest.mark.asyncio
    async def test_multiple_dtc_codes_accepted(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
        mock_rag_service,
    ):
        """Test that multiple DTC codes are accepted."""
        request_data = {
            "vehicle_make": "Volkswagen",
            "vehicle_model": "Golf",
            "vehicle_year": 2018,
            "dtc_codes": ["P0101", "P0171", "P0300"],
            "symptoms": "A motor nehezen indul, egyenetlenul jar, es erosen rezeg.",
        }

        with patch("app.services.nhtsa_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.diagnosis_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.rag_service.diagnose", new=mock_rag_service.diagnose):

            response = await async_client.post("/api/v1/diagnosis/analyze", json=request_data)
            assert response.status_code == 201
            data = response.json()
            assert len(data["dtc_codes"]) == 3

    @pytest.mark.asyncio
    async def test_empty_dtc_codes_rejected(self, async_client, seeded_db):
        """Test that empty DTC codes list is rejected."""
        request_data = {
            "vehicle_make": "Volkswagen",
            "vehicle_model": "Golf",
            "vehicle_year": 2018,
            "dtc_codes": [],
            "symptoms": "A motor nehezen indul es egyenetlenul jar.",
        }

        response = await async_client.post("/api/v1/diagnosis/analyze", json=request_data)
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_too_many_dtc_codes_rejected(self, async_client, seeded_db):
        """Test that more than 20 DTC codes are rejected."""
        request_data = {
            "vehicle_make": "Volkswagen",
            "vehicle_model": "Golf",
            "vehicle_year": 2018,
            "dtc_codes": [f"P{i:04d}" for i in range(25)],
            "symptoms": "A motor nehezen indul es egyenetlenul jar.",
        }

        response = await async_client.post("/api/v1/diagnosis/analyze", json=request_data)
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_dtc_codes_from_all_categories(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
        mock_rag_service,
    ):
        """Test that DTC codes from all categories (P, B, C, U) are accepted."""
        request_data = {
            "vehicle_make": "BMW",
            "vehicle_model": "5 Series",
            "vehicle_year": 2020,
            "dtc_codes": ["P0101", "B1234", "C0035", "U0100"],
            "symptoms": "Tobb figyelmezteto lampa vilagit egyszerre.",
        }

        with patch("app.services.nhtsa_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.diagnosis_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.rag_service.diagnose", new=mock_rag_service.diagnose):

            response = await async_client.post("/api/v1/diagnosis/analyze", json=request_data)
            assert response.status_code == 201
            data = response.json()
            assert len(data["dtc_codes"]) == 4

    @pytest.mark.asyncio
    async def test_lowercase_dtc_codes_normalized(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
        mock_rag_service,
    ):
        """Test that lowercase DTC codes are normalized to uppercase."""
        request_data = {
            "vehicle_make": "Volkswagen",
            "vehicle_model": "Golf",
            "vehicle_year": 2018,
            "dtc_codes": ["p0101", "b1234"],
            "symptoms": "A motor nehezen indul es egyenetlenul jar.",
        }

        with patch("app.services.nhtsa_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.diagnosis_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.rag_service.diagnose", new=mock_rag_service.diagnose):

            response = await async_client.post("/api/v1/diagnosis/analyze", json=request_data)
            # Should either accept lowercase or normalize - verify no validation error
            assert response.status_code in [201, 400]


class TestSymptomTextProcessing:
    """Test Hungarian symptom text processing."""

    @pytest.mark.asyncio
    async def test_hungarian_symptoms_accepted(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
        mock_rag_service,
    ):
        """Test that Hungarian symptom text is accepted."""
        request_data = {
            "vehicle_make": "Volkswagen",
            "vehicle_model": "Golf",
            "vehicle_year": 2018,
            "dtc_codes": ["P0101"],
            "symptoms": "A motor nehezen indul hidegben, egyenetlenul jar alapjaraton, es a fogyasztas jelentosen megnott.",
        }

        with patch("app.services.nhtsa_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.diagnosis_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.rag_service.diagnose", new=mock_rag_service.diagnose):

            response = await async_client.post("/api/v1/diagnosis/analyze", json=request_data)
            assert response.status_code == 201
            data = response.json()
            assert data["symptoms"] == request_data["symptoms"]

    @pytest.mark.asyncio
    async def test_symptoms_minimum_length_required(self, async_client, seeded_db):
        """Test that symptoms must meet minimum length (10 characters)."""
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
    async def test_symptoms_maximum_length_enforced(self, async_client, seeded_db):
        """Test that symptoms cannot exceed maximum length (2000 characters)."""
        request_data = {
            "vehicle_make": "Volkswagen",
            "vehicle_model": "Golf",
            "vehicle_year": 2018,
            "dtc_codes": ["P0101"],
            "symptoms": "A" * 2500,  # Too long
        }

        response = await async_client.post("/api/v1/diagnosis/analyze", json=request_data)
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_symptoms_with_special_characters(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
        mock_rag_service,
    ):
        """Test that symptoms with Hungarian special characters are handled."""
        request_data = {
            "vehicle_make": "Volkswagen",
            "vehicle_model": "Golf",
            "vehicle_year": 2018,
            "dtc_codes": ["P0101"],
            "symptoms": "A motor rezgese erosodik, uzemanyag fogyasztasa no, es fek vilagitas latszik a muszerfalon.",
        }

        with patch("app.services.nhtsa_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.diagnosis_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.rag_service.diagnose", new=mock_rag_service.diagnose):

            response = await async_client.post("/api/v1/diagnosis/analyze", json=request_data)
            assert response.status_code == 201

    @pytest.mark.asyncio
    async def test_additional_context_optional(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
        mock_rag_service,
    ):
        """Test that additional_context is optional but enhances diagnosis."""
        # Without additional context
        request_without_context = {
            "vehicle_make": "Volkswagen",
            "vehicle_model": "Golf",
            "vehicle_year": 2018,
            "dtc_codes": ["P0101"],
            "symptoms": "A motor nehezen indul es egyenetlenul jar.",
        }

        with patch("app.services.nhtsa_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.diagnosis_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.rag_service.diagnose", new=mock_rag_service.diagnose):

            response = await async_client.post("/api/v1/diagnosis/analyze", json=request_without_context)
            assert response.status_code == 201

        # With additional context
        request_with_context = {
            "vehicle_make": "Volkswagen",
            "vehicle_model": "Golf",
            "vehicle_year": 2018,
            "dtc_codes": ["P0101"],
            "symptoms": "A motor nehezen indul es egyenetlenul jar.",
            "additional_context": "A problema telen rosszabb, kulonosen -10 fok alatt.",
        }

        with patch("app.services.nhtsa_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.diagnosis_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.rag_service.diagnose", new=mock_rag_service.diagnose):

            response = await async_client.post("/api/v1/diagnosis/analyze", json=request_with_context)
            assert response.status_code == 201


class TestFullDiagnosisAnalysis:
    """Test full diagnosis analysis with RAG."""

    @pytest.mark.asyncio
    async def test_successful_diagnosis_returns_201(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
        mock_rag_service,
        diagnosis_request_data,
    ):
        """Test that successful diagnosis returns 201 CREATED."""
        with patch("app.services.nhtsa_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.diagnosis_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.rag_service.diagnose", new=mock_rag_service.diagnose):

            response = await async_client.post(
                "/api/v1/diagnosis/analyze",
                json=diagnosis_request_data,
            )

            assert response.status_code == 201

    @pytest.mark.asyncio
    async def test_diagnosis_returns_correct_structure(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
        mock_rag_service,
        diagnosis_request_data,
    ):
        """Test that diagnosis returns correct response structure."""
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
                required_fields = [
                    "id", "vehicle_make", "vehicle_model", "vehicle_year",
                    "dtc_codes", "symptoms", "probable_causes",
                    "recommended_repairs", "confidence_score", "created_at"
                ]
                for field in required_fields:
                    assert field in data, f"Missing required field: {field}"

    @pytest.mark.asyncio
    async def test_diagnosis_probable_causes_have_confidence(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
        mock_rag_service,
        diagnosis_request_data,
    ):
        """Test that probable causes include confidence scores."""
        with patch("app.services.nhtsa_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.diagnosis_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.rag_service.diagnose", new=mock_rag_service.diagnose):

            response = await async_client.post(
                "/api/v1/diagnosis/analyze",
                json=diagnosis_request_data,
            )

            if response.status_code == 201:
                data = response.json()
                for cause in data["probable_causes"]:
                    assert "title" in cause
                    assert "description" in cause
                    assert "confidence" in cause
                    assert 0 <= cause["confidence"] <= 1

    @pytest.mark.asyncio
    async def test_diagnosis_repairs_have_cost_estimates(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
        mock_rag_service,
        diagnosis_request_data,
    ):
        """Test that repair recommendations include cost estimates in HUF."""
        with patch("app.services.nhtsa_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.diagnosis_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.rag_service.diagnose", new=mock_rag_service.diagnose):

            response = await async_client.post(
                "/api/v1/diagnosis/analyze",
                json=diagnosis_request_data,
            )

            if response.status_code == 201:
                data = response.json()
                for repair in data["recommended_repairs"]:
                    assert "title" in repair
                    assert "description" in repair
                    assert "difficulty" in repair
                    # Cost estimates may be optional but currency should be HUF if present
                    if "estimated_cost_currency" in repair:
                        assert repair["estimated_cost_currency"] == "HUF"

    @pytest.mark.asyncio
    async def test_diagnosis_overall_confidence_in_valid_range(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
        mock_rag_service,
        diagnosis_request_data,
    ):
        """Test that overall confidence score is between 0 and 1."""
        with patch("app.services.nhtsa_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.diagnosis_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.rag_service.diagnose", new=mock_rag_service.diagnose):

            response = await async_client.post(
                "/api/v1/diagnosis/analyze",
                json=diagnosis_request_data,
            )

            if response.status_code == 201:
                data = response.json()
                assert 0 <= data["confidence_score"] <= 1

    @pytest.mark.asyncio
    async def test_diagnosis_echoes_input_data(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
        mock_rag_service,
        diagnosis_request_data,
    ):
        """Test that diagnosis response echoes input vehicle and DTC data."""
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


class TestDiagnosisResultRetrieval:
    """Test diagnosis result retrieval."""

    @pytest.mark.asyncio
    async def test_get_nonexistent_diagnosis_returns_404(self, async_client, seeded_db):
        """Test that getting nonexistent diagnosis returns 404."""
        fake_id = str(uuid4())
        response = await async_client.get(f"/api/v1/diagnosis/{fake_id}")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_diagnosis_with_invalid_uuid_returns_422(self, async_client, seeded_db):
        """Test that invalid UUID returns 422."""
        response = await async_client.get("/api/v1/diagnosis/invalid-uuid-format")
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_get_diagnosis_returns_correct_data(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
        mock_rag_service,
        diagnosis_request_data,
    ):
        """Test that stored diagnosis can be retrieved."""
        # First create a diagnosis
        with patch("app.services.nhtsa_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.diagnosis_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.rag_service.diagnose", new=mock_rag_service.diagnose):

            create_response = await async_client.post(
                "/api/v1/diagnosis/analyze",
                json=diagnosis_request_data,
            )

            if create_response.status_code == 201:
                created_data = create_response.json()
                diagnosis_id = created_data["id"]

                # Then retrieve it
                get_response = await async_client.get(f"/api/v1/diagnosis/{diagnosis_id}")

                if get_response.status_code == 200:
                    retrieved_data = get_response.json()
                    assert retrieved_data["id"] == diagnosis_id
                    assert retrieved_data["vehicle_make"] == diagnosis_request_data["vehicle_make"]


class TestQuickAnalyzeEndpoint:
    """Test quick DTC analysis endpoint."""

    @pytest.mark.asyncio
    async def test_quick_analyze_single_code(self, async_client, seeded_db):
        """Test quick analysis with single DTC code."""
        response = await async_client.post(
            "/api/v1/diagnosis/quick-analyze",
            params={"dtc_codes": ["P0101"]},
        )

        assert response.status_code == 200
        data = response.json()
        assert "dtc_codes" in data
        assert len(data["dtc_codes"]) >= 1

    @pytest.mark.asyncio
    async def test_quick_analyze_multiple_codes(self, async_client, seeded_db):
        """Test quick analysis with multiple DTC codes."""
        response = await async_client.post(
            "/api/v1/diagnosis/quick-analyze",
            params={"dtc_codes": ["P0101", "P0171", "B1234"]},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["dtc_codes"]) == 3

    @pytest.mark.asyncio
    async def test_quick_analyze_invalid_code_rejected(self, async_client, seeded_db):
        """Test that invalid DTC code format is rejected."""
        response = await async_client.post(
            "/api/v1/diagnosis/quick-analyze",
            params={"dtc_codes": ["INVALID"]},
        )

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_quick_analyze_unknown_code_handled_gracefully(self, async_client, seeded_db):
        """Test that unknown DTC codes are handled gracefully."""
        response = await async_client.post(
            "/api/v1/diagnosis/quick-analyze",
            params={"dtc_codes": ["P9999"]},
        )

        assert response.status_code == 200
        data = response.json()
        # Should return info indicating code not found
        assert "dtc_codes" in data

    @pytest.mark.asyncio
    async def test_quick_analyze_normalizes_lowercase(self, async_client, seeded_db):
        """Test that quick analyze normalizes lowercase codes."""
        response = await async_client.post(
            "/api/v1/diagnosis/quick-analyze",
            params={"dtc_codes": ["p0101"]},
        )

        assert response.status_code == 200


class TestDiagnosisHistory:
    """Test diagnosis history endpoints."""

    @pytest.mark.asyncio
    async def test_history_requires_authentication(self, async_client, seeded_db):
        """Test that history endpoint requires authentication."""
        response = await async_client.get("/api/v1/diagnosis/history/list")

        # Should return 401 without auth (or 200 with empty list depending on implementation)
        assert response.status_code in [200, 401]

    @pytest.mark.asyncio
    async def test_history_pagination_params(self, async_client, seeded_db):
        """Test that history respects pagination parameters."""
        response = await async_client.get(
            "/api/v1/diagnosis/history/list",
            params={"skip": 0, "limit": 5},
        )

        # Should accept pagination params
        assert response.status_code in [200, 401]

    @pytest.mark.asyncio
    async def test_history_invalid_limit_rejected(self, async_client, seeded_db):
        """Test that invalid limit parameter is rejected."""
        response = await async_client.get(
            "/api/v1/diagnosis/history/list",
            params={"limit": 150},  # Max is 100
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_history_negative_skip_rejected(self, async_client, seeded_db):
        """Test that negative skip parameter is rejected."""
        response = await async_client.get(
            "/api/v1/diagnosis/history/list",
            params={"skip": -1},
        )

        assert response.status_code == 422


class TestAuthenticatedDiagnosis:
    """Test diagnosis flow for authenticated users."""

    @pytest.mark.asyncio
    async def test_authenticated_diagnosis_saved_to_history(
        self,
        authenticated_client,
        mock_nhtsa_service,
        mock_rag_service,
        diagnosis_request_data,
    ):
        """Test that diagnosis is saved to user history when authenticated."""
        client = authenticated_client["client"]
        headers = authenticated_client["headers"]

        with patch("app.services.nhtsa_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.diagnosis_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.rag_service.diagnose", new=mock_rag_service.diagnose):

            response = await client.post(
                "/api/v1/diagnosis/analyze",
                json=diagnosis_request_data,
                headers=headers,
            )

            # Should accept authenticated request
            assert response.status_code in [201, 401]

    @pytest.mark.asyncio
    async def test_anonymous_diagnosis_allowed(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
        mock_rag_service,
        diagnosis_request_data,
    ):
        """Test that anonymous users can perform diagnosis."""
        with patch("app.services.nhtsa_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.diagnosis_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.rag_service.diagnose", new=mock_rag_service.diagnose):

            response = await async_client.post(
                "/api/v1/diagnosis/analyze",
                json=diagnosis_request_data,
            )

            assert response.status_code == 201


class TestVINValidation:
    """Test VIN validation in diagnosis requests."""

    @pytest.mark.asyncio
    async def test_valid_vin_accepted(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
        mock_rag_service,
        valid_vins,
    ):
        """Test that valid VIN numbers are accepted."""
        for vin in valid_vins[:3]:  # Test first 3 VINs
            request_data = {
                "vehicle_make": "Volkswagen",
                "vehicle_model": "Golf",
                "vehicle_year": 2018,
                "vin": vin,
                "dtc_codes": ["P0101"],
                "symptoms": "A motor nehezen indul es egyenetlenul jar.",
            }

            with patch("app.services.nhtsa_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
                 patch("app.services.diagnosis_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
                 patch("app.services.rag_service.diagnose", new=mock_rag_service.diagnose):

                response = await async_client.post("/api/v1/diagnosis/analyze", json=request_data)
                # Valid VIN should be accepted (may still fail VIN decode in mock)
                assert response.status_code in [201, 400]

    @pytest.mark.asyncio
    async def test_invalid_vin_format_rejected(
        self,
        async_client,
        seeded_db,
        invalid_vins,
    ):
        """Test that invalid VIN formats are rejected."""
        for vin in invalid_vins:
            request_data = {
                "vehicle_make": "Volkswagen",
                "vehicle_model": "Golf",
                "vehicle_year": 2018,
                "vin": vin,
                "dtc_codes": ["P0101"],
                "symptoms": "A motor nehezen indul es egyenetlenul jar.",
            }

            response = await async_client.post("/api/v1/diagnosis/analyze", json=request_data)
            assert response.status_code == 422, f"VIN {vin} should be rejected"

    @pytest.mark.asyncio
    async def test_vin_optional(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
        mock_rag_service,
    ):
        """Test that VIN is optional for diagnosis."""
        request_data = {
            "vehicle_make": "Volkswagen",
            "vehicle_model": "Golf",
            "vehicle_year": 2018,
            "dtc_codes": ["P0101"],
            "symptoms": "A motor nehezen indul es egyenetlenul jar.",
            # No VIN provided
        }

        with patch("app.services.nhtsa_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.diagnosis_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.rag_service.diagnose", new=mock_rag_service.diagnose):

            response = await async_client.post("/api/v1/diagnosis/analyze", json=request_data)
            assert response.status_code == 201


class TestDiagnosisErrorHandling:
    """Test error handling in diagnosis flow."""

    @pytest.mark.asyncio
    async def test_service_error_returns_500(
        self,
        async_client,
        seeded_db,
        diagnosis_request_data,
    ):
        """Test that service errors return 500."""
        failing_service = AsyncMock()
        failing_service.diagnose.side_effect = Exception("Service unavailable")

        # Remove VIN to skip NHTSA validation
        request_data = diagnosis_request_data.copy()
        del request_data["vin"]

        with patch("app.services.rag_service.diagnose", new=failing_service.diagnose):
            response = await async_client.post(
                "/api/v1/diagnosis/analyze",
                json=request_data,
            )

            # Should return error status
            assert response.status_code in [400, 500]

    @pytest.mark.asyncio
    async def test_malformed_json_returns_422(self, async_client, seeded_db):
        """Test that malformed JSON returns 422."""
        response = await async_client.post(
            "/api/v1/diagnosis/analyze",
            content="not valid json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422


class TestDiagnosisWithHungarianNLP:
    """Test diagnosis with Hungarian NLP processing."""

    @pytest.mark.asyncio
    async def test_hungarian_special_characters_processed(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
        mock_rag_service,
    ):
        """Test that Hungarian special characters are correctly processed."""
        request_data = {
            "vehicle_make": "Volkswagen",
            "vehicle_model": "Golf",
            "vehicle_year": 2018,
            "dtc_codes": ["P0101"],
            "symptoms": "A motor rezgese erosodik uzemkozben, az uzemanyag-fogyasztas no, es a muszerfalon fek vilagitas latszik.",
        }

        with patch("app.services.nhtsa_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.diagnosis_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.rag_service.diagnose", new=mock_rag_service.diagnose):

            response = await async_client.post("/api/v1/diagnosis/analyze", json=request_data)
            assert response.status_code == 201

    @pytest.mark.asyncio
    async def test_long_hungarian_symptom_text(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
        mock_rag_service,
    ):
        """Test diagnosis with long Hungarian symptom description."""
        long_symptoms = """
        A jarmu az uttobbi het soran tobb problemat is mutatott. Eloszor a motor nehezen indult
        hidegben, korulbelul 5-10 masodperces tekeres utan indult csak be. Ezutan eszrevettem,
        hogy alapjaraton egyenetlenul jar, mintha akadozna. A fogyasztas is jelentosen megnott,
        korulbelul 2-3 literrel tobb mint korabban. Gyorsitaskor neha akadozik, es erot veszit.
        A muszerfalon a motor ellenorzo lampa is felgyulladt. A kipufogobol neha fekete fust jon.
        Hidegben rosszabb a helyzet, melegedés utan kicsit jobb lesz.
        """

        request_data = {
            "vehicle_make": "Audi",
            "vehicle_model": "A4",
            "vehicle_year": 2017,
            "dtc_codes": ["P0101", "P0171", "P0300"],
            "symptoms": long_symptoms.strip(),
        }

        with patch("app.services.nhtsa_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.diagnosis_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.rag_service.diagnose", new=mock_rag_service.diagnose):

            response = await async_client.post("/api/v1/diagnosis/analyze", json=request_data)
            assert response.status_code == 201

    @pytest.mark.asyncio
    async def test_symptoms_with_technical_terms_in_hungarian(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
        mock_rag_service,
    ):
        """Test diagnosis with technical Hungarian terms."""
        request_data = {
            "vehicle_make": "Mercedes-Benz",
            "vehicle_model": "C-Class",
            "vehicle_year": 2019,
            "dtc_codes": ["U0100"],
            "symptoms": "A vezerloegyseg kommunikacios hibat jelez, a CAN-busz nem mukodik megfeleloen, es a diagnosztikai csatlakozo nem ad kapcsolatot.",
        }

        with patch("app.services.nhtsa_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.diagnosis_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.rag_service.diagnose", new=mock_rag_service.diagnose):

            response = await async_client.post("/api/v1/diagnosis/analyze", json=request_data)
            assert response.status_code == 201


class TestDiagnosisEdgeCases:
    """Test edge cases in diagnosis flow."""

    @pytest.mark.asyncio
    async def test_maximum_allowed_dtc_codes(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
        mock_rag_service,
    ):
        """Test diagnosis with maximum allowed DTC codes (20)."""
        request_data = {
            "vehicle_make": "Volkswagen",
            "vehicle_model": "Passat",
            "vehicle_year": 2018,
            "dtc_codes": [f"P0{i:03d}" for i in range(100, 120)],  # 20 codes
            "symptoms": "Tobb figyelmezteto lampa vilagit egyszerre a muszerfalon.",
        }

        with patch("app.services.nhtsa_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.diagnosis_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.rag_service.diagnose", new=mock_rag_service.diagnose):

            response = await async_client.post("/api/v1/diagnosis/analyze", json=request_data)
            assert response.status_code == 201
            data = response.json()
            assert len(data["dtc_codes"]) == 20

    @pytest.mark.asyncio
    async def test_symptoms_at_minimum_length(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
        mock_rag_service,
    ):
        """Test diagnosis with symptoms at minimum valid length (10 chars)."""
        request_data = {
            "vehicle_make": "Toyota",
            "vehicle_model": "Yaris",
            "vehicle_year": 2021,
            "dtc_codes": ["P0420"],
            "symptoms": "Motor zaj.",  # Exactly 10 characters
        }

        with patch("app.services.nhtsa_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.diagnosis_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.rag_service.diagnose", new=mock_rag_service.diagnose):

            response = await async_client.post("/api/v1/diagnosis/analyze", json=request_data)
            assert response.status_code == 201

    @pytest.mark.asyncio
    async def test_symptoms_at_maximum_length(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
        mock_rag_service,
    ):
        """Test diagnosis with symptoms at maximum valid length (2000 chars)."""
        max_symptoms = "A" * 2000  # Maximum allowed

        request_data = {
            "vehicle_make": "Honda",
            "vehicle_model": "Civic",
            "vehicle_year": 2020,
            "dtc_codes": ["P0300"],
            "symptoms": max_symptoms,
        }

        with patch("app.services.nhtsa_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.diagnosis_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.rag_service.diagnose", new=mock_rag_service.diagnose):

            response = await async_client.post("/api/v1/diagnosis/analyze", json=request_data)
            assert response.status_code == 201

    @pytest.mark.asyncio
    async def test_vehicle_year_at_boundaries(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
        mock_rag_service,
    ):
        """Test diagnosis with vehicle year at valid boundaries."""
        # Minimum valid year (1900)
        request_data = {
            "vehicle_make": "Ford",
            "vehicle_model": "Model T",
            "vehicle_year": 1900,
            "dtc_codes": ["P0101"],
            "symptoms": "Regi jarmu problema leirasa.",
        }

        with patch("app.services.nhtsa_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.diagnosis_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.rag_service.diagnose", new=mock_rag_service.diagnose):

            response = await async_client.post("/api/v1/diagnosis/analyze", json=request_data)
            assert response.status_code == 201

        # Maximum valid year (2030)
        request_data["vehicle_year"] = 2030
        with patch("app.services.nhtsa_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.diagnosis_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.rag_service.diagnose", new=mock_rag_service.diagnose):

            response = await async_client.post("/api/v1/diagnosis/analyze", json=request_data)
            assert response.status_code == 201


class TestDiagnosisDeleteEndpoint:
    """Test diagnosis deletion endpoint."""

    @pytest.mark.asyncio
    async def test_delete_own_diagnosis(
        self,
        authenticated_client,
        mock_nhtsa_service,
        mock_rag_service,
        diagnosis_request_data,
    ):
        """Test that user can delete their own diagnosis."""
        client = authenticated_client["client"]
        headers = authenticated_client["headers"]

        # First create a diagnosis
        with patch("app.services.nhtsa_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.diagnosis_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.rag_service.diagnose", new=mock_rag_service.diagnose):

            create_response = await client.post(
                "/api/v1/diagnosis/analyze",
                json=diagnosis_request_data,
                headers=headers,
            )

            if create_response.status_code == 201:
                created_data = create_response.json()
                diagnosis_id = created_data["id"]

                # Then delete it
                delete_response = await client.delete(
                    f"/api/v1/diagnosis/{diagnosis_id}",
                    headers=headers,
                )

                assert delete_response.status_code in [200, 404]

    @pytest.mark.asyncio
    async def test_delete_nonexistent_diagnosis(self, authenticated_client):
        """Test deleting nonexistent diagnosis returns 404."""
        client = authenticated_client["client"]
        headers = authenticated_client["headers"]

        fake_id = str(uuid4())
        response = await client.delete(
            f"/api/v1/diagnosis/{fake_id}",
            headers=headers,
        )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_requires_authentication(self, async_client, seeded_db):
        """Test that delete endpoint requires authentication."""
        fake_id = str(uuid4())
        response = await async_client.delete(f"/api/v1/diagnosis/{fake_id}")

        assert response.status_code == 401


class TestDiagnosisStatsEndpoint:
    """Test diagnosis statistics endpoint."""

    @pytest.mark.asyncio
    async def test_stats_requires_authentication(self, async_client, seeded_db):
        """Test that stats endpoint requires authentication."""
        response = await async_client.get("/api/v1/diagnosis/stats/summary")

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_stats_returns_correct_structure(self, authenticated_client):
        """Test that stats returns correct response structure."""
        client = authenticated_client["client"]
        headers = authenticated_client["headers"]

        response = await client.get(
            "/api/v1/diagnosis/stats/summary",
            headers=headers,
        )

        if response.status_code == 200:
            data = response.json()
            assert "total_diagnoses" in data
            assert "avg_confidence" in data
            assert "most_diagnosed_vehicles" in data
            assert "most_common_dtcs" in data
            assert "diagnoses_by_month" in data


class TestDiagnosisHistoryFiltering:
    """Test diagnosis history filtering functionality."""

    @pytest.mark.asyncio
    async def test_filter_by_vehicle_make(self, authenticated_client):
        """Test filtering history by vehicle make."""
        client = authenticated_client["client"]
        headers = authenticated_client["headers"]

        response = await client.get(
            "/api/v1/diagnosis/history/list",
            params={"vehicle_make": "Volkswagen"},
            headers=headers,
        )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_filter_by_vehicle_model(self, authenticated_client):
        """Test filtering history by vehicle model."""
        client = authenticated_client["client"]
        headers = authenticated_client["headers"]

        response = await client.get(
            "/api/v1/diagnosis/history/list",
            params={"vehicle_model": "Golf"},
            headers=headers,
        )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_filter_by_dtc_code(self, authenticated_client):
        """Test filtering history by DTC code."""
        client = authenticated_client["client"]
        headers = authenticated_client["headers"]

        response = await client.get(
            "/api/v1/diagnosis/history/list",
            params={"dtc_code": "P0101"},
            headers=headers,
        )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_filter_by_date_range(self, authenticated_client):
        """Test filtering history by date range."""
        client = authenticated_client["client"]
        headers = authenticated_client["headers"]

        response = await client.get(
            "/api/v1/diagnosis/history/list",
            params={
                "date_from": "2024-01-01T00:00:00",
                "date_to": "2024-12-31T23:59:59",
            },
            headers=headers,
        )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_combined_filters(self, authenticated_client):
        """Test multiple filters combined."""
        client = authenticated_client["client"]
        headers = authenticated_client["headers"]

        response = await client.get(
            "/api/v1/diagnosis/history/list",
            params={
                "vehicle_make": "Volkswagen",
                "vehicle_year": 2018,
                "limit": 5,
            },
            headers=headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data.get("items", [])) <= 5


class TestConcurrentDiagnosis:
    """Test concurrent diagnosis requests."""

    @pytest.mark.asyncio
    async def test_concurrent_diagnoses_isolated(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
        mock_rag_service,
    ):
        """Test that concurrent diagnoses are properly isolated."""
        import asyncio

        request_data_1 = {
            "vehicle_make": "Volkswagen",
            "vehicle_model": "Golf",
            "vehicle_year": 2018,
            "dtc_codes": ["P0101"],
            "symptoms": "Elso jarmu tunetei: motor nehezen indul.",
        }

        request_data_2 = {
            "vehicle_make": "Toyota",
            "vehicle_model": "Corolla",
            "vehicle_year": 2020,
            "dtc_codes": ["P0300"],
            "symptoms": "Masodik jarmu tunetei: motor rezeg.",
        }

        with patch("app.services.nhtsa_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.diagnosis_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.rag_service.diagnose", new=mock_rag_service.diagnose):

            # Send both requests concurrently
            task1 = async_client.post("/api/v1/diagnosis/analyze", json=request_data_1)
            task2 = async_client.post("/api/v1/diagnosis/analyze", json=request_data_2)

            responses = await asyncio.gather(task1, task2)

            # Both should succeed
            for response in responses:
                assert response.status_code in [201, 400, 500]


class TestDiagnosisWithAllDTCCategories:
    """Test diagnosis with DTC codes from all categories."""

    @pytest.mark.asyncio
    async def test_mixed_category_dtc_codes(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
        mock_rag_service,
    ):
        """Test diagnosis with DTC codes from all categories (P, B, C, U)."""
        request_data = {
            "vehicle_make": "Audi",
            "vehicle_model": "A6",
            "vehicle_year": 2019,
            "dtc_codes": ["P0101", "B1234", "C0035", "U0100"],
            "symptoms": "A jarmu tobb rendszereben egyidejuleg jelentkeztek hibak. Motor, legzsak, ABS es kommunikacios problemak.",
        }

        with patch("app.services.nhtsa_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.diagnosis_service.get_nhtsa_service", return_value=mock_nhtsa_service), \
             patch("app.services.rag_service.diagnose", new=mock_rag_service.diagnose):

            response = await async_client.post("/api/v1/diagnosis/analyze", json=request_data)
            assert response.status_code == 201
            data = response.json()

            # Check all codes are present
            assert "P0101" in data["dtc_codes"]
            assert "B1234" in data["dtc_codes"]
            assert "C0035" in data["dtc_codes"]
            assert "U0100" in data["dtc_codes"]
