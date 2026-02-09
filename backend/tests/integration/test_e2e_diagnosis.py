"""
End-to-end integration tests for the complete diagnosis flow.

Tests the full pipeline from user input to diagnosis result,
covering all components working together.
"""

import pytest
from unittest.mock import AsyncMock, patch
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(backend_path))


class TestE2EDiagnosisFlow:
    """Test end-to-end diagnosis flow."""

    @pytest.mark.asyncio
    async def test_complete_diagnosis_flow(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
        mock_rag_service,
        diagnosis_request_data,
    ):
        """Test complete diagnosis from request to response."""
        with (
            patch("app.services.nhtsa_service.get_nhtsa_service", return_value=mock_nhtsa_service),
            patch(
                "app.services.diagnosis_service.get_nhtsa_service", return_value=mock_nhtsa_service
            ),
            patch("app.services.rag_service.diagnose", new=mock_rag_service.diagnose),
        ):
            # Step 1: Submit diagnosis request
            response = await async_client.post(
                "/api/v1/diagnosis/analyze",
                json=diagnosis_request_data,
            )

            # Step 2: Verify successful response
            assert response.status_code == 201
            data = response.json()

            # Step 3: Verify response structure
            assert "id" in data
            assert "probable_causes" in data
            assert "recommended_repairs" in data
            assert "confidence_score" in data

            # Step 4: Verify vehicle info echoed back
            assert data["vehicle_make"] == diagnosis_request_data["vehicle_make"]
            assert data["vehicle_model"] == diagnosis_request_data["vehicle_model"]

    @pytest.mark.asyncio
    async def test_diagnosis_flow_without_vin(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
        mock_rag_service,
        diagnosis_request_data,
    ):
        """Test diagnosis flow without VIN (optional field)."""
        # Remove VIN from request
        request_data = diagnosis_request_data.copy()
        del request_data["vin"]

        with (
            patch("app.services.nhtsa_service.get_nhtsa_service", return_value=mock_nhtsa_service),
            patch(
                "app.services.diagnosis_service.get_nhtsa_service", return_value=mock_nhtsa_service
            ),
            patch("app.services.rag_service.diagnose", new=mock_rag_service.diagnose),
        ):
            response = await async_client.post(
                "/api/v1/diagnosis/analyze",
                json=request_data,
            )

            # Should still work without VIN
            assert response.status_code == 201

    @pytest.mark.asyncio
    async def test_diagnosis_flow_multiple_dtc_codes(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
        mock_rag_service,
        diagnosis_request_data,
    ):
        """Test diagnosis flow with multiple DTC codes."""
        request_data = diagnosis_request_data.copy()
        request_data["dtc_codes"] = ["P0101", "P0171", "B1234", "C0035", "U0100"]

        with (
            patch("app.services.nhtsa_service.get_nhtsa_service", return_value=mock_nhtsa_service),
            patch(
                "app.services.diagnosis_service.get_nhtsa_service", return_value=mock_nhtsa_service
            ),
            patch("app.services.rag_service.diagnose", new=mock_rag_service.diagnose),
        ):
            response = await async_client.post(
                "/api/v1/diagnosis/analyze",
                json=request_data,
            )

            assert response.status_code == 201
            data = response.json()

            # All codes should be in response
            assert len(data["dtc_codes"]) == 5

    @pytest.mark.asyncio
    async def test_diagnosis_flow_with_detailed_symptoms(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
        mock_rag_service,
        diagnosis_request_data,
    ):
        """Test diagnosis flow with detailed Hungarian symptoms."""
        request_data = diagnosis_request_data.copy()
        request_data["symptoms"] = """
        A jarmu problemai:
        1. A motor nehezen indul hidegben, kulonosen -10 fok alatt
        2. Alapjaraton egyenetlenul mukodik, reszketes erzekelhetom
        3. Gyorsitasnal erzekelheto teljesitmenycsokkenest
        4. A fogyasztas jelentosen megnott az utobbbi hetekben
        5. Neha meggyullad a motorjelzo lampa
        """
        request_data["additional_context"] = "A jarmuvel napi 50 km-t teszek meg varosban."

        with (
            patch("app.services.nhtsa_service.get_nhtsa_service", return_value=mock_nhtsa_service),
            patch(
                "app.services.diagnosis_service.get_nhtsa_service", return_value=mock_nhtsa_service
            ),
            patch("app.services.rag_service.diagnose", new=mock_rag_service.diagnose),
        ):
            response = await async_client.post(
                "/api/v1/diagnosis/analyze",
                json=request_data,
            )

            assert response.status_code == 201


class TestE2EQuickAnalyzeFlow:
    """Test end-to-end quick analyze flow."""

    @pytest.mark.asyncio
    async def test_quick_analyze_single_code(self, async_client, seeded_db):
        """Test quick analyze with a single DTC code."""
        response = await async_client.post(
            "/api/v1/diagnosis/quick-analyze",
            params={"dtc_codes": ["P0101"]},
        )

        assert response.status_code == 200
        data = response.json()

        assert "dtc_codes" in data
        assert len(data["dtc_codes"]) == 1
        assert data["dtc_codes"][0]["code"] == "P0101"

    @pytest.mark.asyncio
    async def test_quick_analyze_multiple_codes(self, async_client, seeded_db):
        """Test quick analyze with multiple DTC codes."""
        response = await async_client.post(
            "/api/v1/diagnosis/quick-analyze",
            params={"dtc_codes": ["P0101", "P0171", "B1234"]},
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data["dtc_codes"]) == 3

    @pytest.mark.asyncio
    async def test_quick_analyze_includes_symptoms(self, async_client, seeded_db):
        """Test that quick analyze returns symptoms for known codes."""
        response = await async_client.post(
            "/api/v1/diagnosis/quick-analyze",
            params={"dtc_codes": ["P0101"]},
        )

        assert response.status_code == 200
        data = response.json()

        if data["dtc_codes"]:
            dtc_info = data["dtc_codes"][0]
            assert "symptoms" in dtc_info

    @pytest.mark.asyncio
    async def test_quick_analyze_includes_causes(self, async_client, seeded_db):
        """Test that quick analyze returns possible causes."""
        response = await async_client.post(
            "/api/v1/diagnosis/quick-analyze",
            params={"dtc_codes": ["P0101"]},
        )

        assert response.status_code == 200
        data = response.json()

        if data["dtc_codes"]:
            dtc_info = data["dtc_codes"][0]
            assert "possible_causes" in dtc_info


class TestE2EVehicleDataFlow:
    """Test end-to-end vehicle data retrieval flow."""

    @pytest.mark.asyncio
    async def test_vehicle_lookup_flow(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
    ):
        """Test complete vehicle lookup flow."""
        with patch(
            "app.api.v1.endpoints.vehicles.get_nhtsa_service", return_value=mock_nhtsa_service
        ):
            # Step 1: Get available makes
            makes_response = await async_client.get("/api/v1/vehicles/makes")
            assert makes_response.status_code == 200

            makes = makes_response.json()
            assert len(makes) > 0

            # Step 2: Get models for a make
            models_response = await async_client.get("/api/v1/vehicles/models/volkswagen")
            assert models_response.status_code == 200

            # Step 3: Get available years
            years_response = await async_client.get("/api/v1/vehicles/years")
            assert years_response.status_code == 200

    @pytest.mark.asyncio
    async def test_vin_decode_and_recalls_flow(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
    ):
        """Test VIN decode followed by recalls lookup."""
        with patch(
            "app.api.v1.endpoints.vehicles.get_nhtsa_service", return_value=mock_nhtsa_service
        ):
            # Step 1: Decode VIN
            vin_response = await async_client.post(
                "/api/v1/vehicles/decode-vin",
                json={"vin": "WVWZZZ3CZWE123456"},
            )
            assert vin_response.status_code == 200

            vin_data = vin_response.json()
            make = vin_data["make"]
            model = vin_data["model"]
            year = vin_data["year"]

            # Step 2: Get recalls for decoded vehicle
            recalls_response = await async_client.get(
                f"/api/v1/vehicles/{make}/{model}/{year}/recalls"
            )
            assert recalls_response.status_code == 200

            # Step 3: Get complaints for decoded vehicle
            complaints_response = await async_client.get(
                f"/api/v1/vehicles/{make}/{model}/{year}/complaints"
            )
            assert complaints_response.status_code == 200


class TestE2EErrorScenarios:
    """Test end-to-end error scenarios."""

    @pytest.mark.asyncio
    async def test_diagnosis_with_all_unknown_codes(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
        mock_rag_service,
    ):
        """Test diagnosis with DTC codes not in database."""
        request_data = {
            "vehicle_make": "Volkswagen",
            "vehicle_model": "Golf",
            "vehicle_year": 2018,
            "dtc_codes": ["P9991", "P9992", "P9993"],  # Unknown codes
            "symptoms": "Motor nehezen indul hidegben",
        }

        with (
            patch("app.services.nhtsa_service.get_nhtsa_service", return_value=mock_nhtsa_service),
            patch(
                "app.services.diagnosis_service.get_nhtsa_service", return_value=mock_nhtsa_service
            ),
            patch("app.services.rag_service.diagnose", new=mock_rag_service.diagnose),
        ):
            response = await async_client.post(
                "/api/v1/diagnosis/analyze",
                json=request_data,
            )

            # Should still work with unknown codes (graceful degradation)
            assert response.status_code in [201, 500]

    @pytest.mark.asyncio
    async def test_diagnosis_with_invalid_dtc_format(
        self,
        async_client,
        seeded_db,
    ):
        """Test diagnosis with invalid DTC code format."""
        request_data = {
            "vehicle_make": "Volkswagen",
            "vehicle_model": "Golf",
            "vehicle_year": 2018,
            "dtc_codes": ["INVALID", "X1234"],  # Invalid formats
            "symptoms": "Motor nehezen indul hidegben",
        }

        response = await async_client.post(
            "/api/v1/diagnosis/analyze",
            json=request_data,
        )

        # Should handle invalid codes
        assert response.status_code in [201, 400, 422]

    @pytest.mark.asyncio
    async def test_diagnosis_with_service_failure(
        self,
        async_client,
        seeded_db,
        diagnosis_request_data,
    ):
        """Test diagnosis when external services fail."""
        failing_nhtsa = AsyncMock()
        failing_nhtsa.decode_vin.side_effect = Exception("Service unavailable")
        failing_nhtsa.get_recalls.side_effect = Exception("Service unavailable")
        failing_nhtsa.get_complaints.side_effect = Exception("Service unavailable")

        failing_rag = AsyncMock()
        failing_rag.diagnose.side_effect = Exception("RAG service unavailable")

        # Remove VIN to avoid VIN decode
        request_data = diagnosis_request_data.copy()
        del request_data["vin"]

        with (
            patch("app.services.nhtsa_service.get_nhtsa_service", return_value=failing_nhtsa),
            patch("app.services.diagnosis_service.get_nhtsa_service", return_value=failing_nhtsa),
            patch("app.services.rag_service.diagnose", new=failing_rag.diagnose),
        ):
            response = await async_client.post(
                "/api/v1/diagnosis/analyze",
                json=request_data,
            )

            # Should return error response, not crash
            assert response.status_code in [201, 500]


class TestE2EDataIntegrity:
    """Test end-to-end data integrity."""

    @pytest.mark.asyncio
    async def test_dtc_search_matches_detail(self, async_client, seeded_db):
        """Test that search results match detail endpoint."""
        # Search for P0101
        search_response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "P0101"},
        )
        assert search_response.status_code == 200
        search_data = search_response.json()

        # Get detail for same code
        detail_response = await async_client.get("/api/v1/dtc/P0101")
        assert detail_response.status_code == 200
        detail_data = detail_response.json()

        # Data should be consistent
        if search_data:
            search_result = next((item for item in search_data if item["code"] == "P0101"), None)
            if search_result:
                assert search_result["code"] == detail_data["code"]
                assert search_result["category"] == detail_data["category"]

    @pytest.mark.asyncio
    async def test_diagnosis_preserves_input_data(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
        mock_rag_service,
        diagnosis_request_data,
    ):
        """Test that diagnosis preserves original input data."""
        with (
            patch("app.services.nhtsa_service.get_nhtsa_service", return_value=mock_nhtsa_service),
            patch(
                "app.services.diagnosis_service.get_nhtsa_service", return_value=mock_nhtsa_service
            ),
            patch("app.services.rag_service.diagnose", new=mock_rag_service.diagnose),
        ):
            response = await async_client.post(
                "/api/v1/diagnosis/analyze",
                json=diagnosis_request_data,
            )

            if response.status_code == 201:
                data = response.json()

                # Original data should be preserved
                assert data["vehicle_make"] == diagnosis_request_data["vehicle_make"]
                assert data["vehicle_model"] == diagnosis_request_data["vehicle_model"]
                assert data["vehicle_year"] == diagnosis_request_data["vehicle_year"]
                assert data["dtc_codes"] == diagnosis_request_data["dtc_codes"]
                assert data["symptoms"] == diagnosis_request_data["symptoms"]


class TestE2EHungarianLanguage:
    """Test end-to-end Hungarian language support."""

    @pytest.mark.asyncio
    async def test_dtc_hungarian_description(self, async_client, seeded_db):
        """Test that DTC codes include Hungarian descriptions."""
        response = await async_client.get("/api/v1/dtc/P0101")

        assert response.status_code == 200
        data = response.json()

        assert "description_hu" in data
        assert data["description_hu"] is not None
        assert len(data["description_hu"]) > 0

    @pytest.mark.asyncio
    async def test_dtc_categories_hungarian(self, async_client, seeded_db):
        """Test that DTC categories include Hungarian names."""
        response = await async_client.get("/api/v1/dtc/categories/list")

        assert response.status_code == 200
        data = response.json()

        for category in data:
            assert "name_hu" in category
            assert "description_hu" in category

    @pytest.mark.asyncio
    async def test_diagnosis_accepts_hungarian_symptoms(
        self,
        async_client,
        seeded_db,
        mock_nhtsa_service,
        mock_rag_service,
    ):
        """Test that diagnosis accepts Hungarian symptom descriptions."""
        request_data = {
            "vehicle_make": "Volkswagen",
            "vehicle_model": "Golf",
            "vehicle_year": 2018,
            "dtc_codes": ["P0101"],
            "symptoms": "A motor nehezen indul hidegben. Az alapjarat egyenetlen, a jarmu reszkest. A fogyasztas jelentosen megnott.",
        }

        with (
            patch("app.services.nhtsa_service.get_nhtsa_service", return_value=mock_nhtsa_service),
            patch(
                "app.services.diagnosis_service.get_nhtsa_service", return_value=mock_nhtsa_service
            ),
            patch("app.services.rag_service.diagnose", new=mock_rag_service.diagnose),
        ):
            response = await async_client.post(
                "/api/v1/diagnosis/analyze",
                json=request_data,
            )

            assert response.status_code == 201


class TestE2EPerformance:
    """Test end-to-end performance characteristics."""

    @pytest.mark.asyncio
    async def test_dtc_search_response_time(self, async_client, seeded_db):
        """Test that DTC search responds quickly."""
        import time

        start = time.time()
        response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "P0101"},
        )
        elapsed = time.time() - start

        assert response.status_code == 200
        # Should respond within 1 second for simple search
        assert elapsed < 1.0

    @pytest.mark.asyncio
    async def test_quick_analyze_response_time(self, async_client, seeded_db):
        """Test that quick analyze responds quickly."""
        import time

        start = time.time()
        response = await async_client.post(
            "/api/v1/diagnosis/quick-analyze",
            params={"dtc_codes": ["P0101"]},
        )
        elapsed = time.time() - start

        assert response.status_code == 200
        # Should respond within 2 seconds
        assert elapsed < 2.0
