"""
API tests for diagnosis endpoints.

Tests:
- POST /api/v1/diagnosis/analyze - Main diagnosis endpoint
- POST /api/v1/diagnosis/quick-analyze - Quick DTC lookup
- GET /api/v1/diagnosis/{diagnosis_id} - Get diagnosis by ID
- GET /api/v1/diagnosis/history/list - Get diagnosis history
- DELETE /api/v1/diagnosis/{diagnosis_id} - Delete diagnosis
- GET /api/v1/diagnosis/stats/summary - Get diagnosis statistics
"""

import pytest
from httpx import AsyncClient
from unittest.mock import patch, AsyncMock, MagicMock
from uuid import uuid4


class TestDiagnosisAnalyze:
    """Tests for POST /api/v1/diagnosis/analyze endpoint."""

    @pytest.mark.asyncio
    async def test_analyze_requires_vehicle_make(
        self, async_client: AsyncClient, diagnosis_request_data: dict
    ):
        """Test that analyze requires vehicle_make."""
        data = diagnosis_request_data.copy()
        del data["vehicle_make"]

        response = await async_client.post("/api/v1/diagnosis/analyze", json=data)

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_analyze_requires_vehicle_model(
        self, async_client: AsyncClient, diagnosis_request_data: dict
    ):
        """Test that analyze requires vehicle_model."""
        data = diagnosis_request_data.copy()
        del data["vehicle_model"]

        response = await async_client.post("/api/v1/diagnosis/analyze", json=data)

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_analyze_requires_vehicle_year(
        self, async_client: AsyncClient, diagnosis_request_data: dict
    ):
        """Test that analyze requires vehicle_year."""
        data = diagnosis_request_data.copy()
        del data["vehicle_year"]

        response = await async_client.post("/api/v1/diagnosis/analyze", json=data)

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_analyze_requires_dtc_codes(
        self, async_client: AsyncClient, diagnosis_request_data: dict
    ):
        """Test that analyze requires dtc_codes."""
        data = diagnosis_request_data.copy()
        del data["dtc_codes"]

        response = await async_client.post("/api/v1/diagnosis/analyze", json=data)

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_analyze_requires_symptoms(
        self, async_client: AsyncClient, diagnosis_request_data: dict
    ):
        """Test that analyze requires symptoms."""
        data = diagnosis_request_data.copy()
        del data["symptoms"]

        response = await async_client.post("/api/v1/diagnosis/analyze", json=data)

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_analyze_validates_year_range(
        self, async_client: AsyncClient, diagnosis_request_data: dict
    ):
        """Test that analyze validates year range."""
        data = diagnosis_request_data.copy()
        data["vehicle_year"] = 1800  # Before 1900

        response = await async_client.post("/api/v1/diagnosis/analyze", json=data)

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_analyze_validates_symptoms_min_length(
        self, async_client: AsyncClient, diagnosis_request_data: dict
    ):
        """Test that analyze validates symptoms minimum length."""
        data = diagnosis_request_data.copy()
        data["symptoms"] = "short"  # Less than 10 characters

        response = await async_client.post("/api/v1/diagnosis/analyze", json=data)

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_analyze_validates_dtc_codes_not_empty(
        self, async_client: AsyncClient, diagnosis_request_data: dict
    ):
        """Test that analyze validates dtc_codes is not empty."""
        data = diagnosis_request_data.copy()
        data["dtc_codes"] = []

        response = await async_client.post("/api/v1/diagnosis/analyze", json=data)

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_analyze_validates_vin_length(
        self, async_client: AsyncClient, diagnosis_request_data: dict
    ):
        """Test that analyze validates VIN length if provided."""
        data = diagnosis_request_data.copy()
        data["vin"] = "SHORT"  # Not 17 characters

        response = await async_client.post("/api/v1/diagnosis/analyze", json=data)

        assert response.status_code == 422


class TestQuickAnalyze:
    """Tests for POST /api/v1/diagnosis/quick-analyze endpoint."""

    @pytest.mark.asyncio
    async def test_quick_analyze_returns_200(self, async_client: AsyncClient, sample_dtc_codes):
        """Test quick analyze returns 200."""
        response = await async_client.post(
            "/api/v1/diagnosis/quick-analyze",
            params={"dtc_codes": ["P0101"]},
        )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_quick_analyze_returns_dtc_info(
        self, async_client: AsyncClient, sample_dtc_codes
    ):
        """Test quick analyze returns DTC information."""
        response = await async_client.post(
            "/api/v1/diagnosis/quick-analyze",
            params={"dtc_codes": ["P0101"]},
        )

        assert response.status_code == 200
        data = response.json()

        assert "dtc_codes" in data
        assert isinstance(data["dtc_codes"], list)
        assert "message" in data

    @pytest.mark.asyncio
    async def test_quick_analyze_multiple_codes(self, async_client: AsyncClient, sample_dtc_codes):
        """Test quick analyze with multiple codes."""
        response = await async_client.post(
            "/api/v1/diagnosis/quick-analyze",
            params={"dtc_codes": ["P0101", "P0171"]},
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data["dtc_codes"]) == 2

    @pytest.mark.asyncio
    async def test_quick_analyze_invalid_code_format_returns_400(self, async_client: AsyncClient):
        """Test quick analyze with invalid DTC format returns 400."""
        response = await async_client.post(
            "/api/v1/diagnosis/quick-analyze",
            params={"dtc_codes": ["INVALID"]},
        )

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_quick_analyze_unknown_code_returns_placeholder(
        self, async_client: AsyncClient, sample_dtc_codes
    ):
        """Test quick analyze with unknown code returns placeholder."""
        response = await async_client.post(
            "/api/v1/diagnosis/quick-analyze",
            params={"dtc_codes": ["P9999"]},  # Not in database
        )

        assert response.status_code == 200
        data = response.json()

        # Should still return something for unknown codes
        assert len(data["dtc_codes"]) == 1
        assert data["dtc_codes"][0]["code"] == "P9999"

    @pytest.mark.asyncio
    async def test_quick_analyze_missing_codes_returns_422(self, async_client: AsyncClient):
        """Test quick analyze without codes returns 422."""
        response = await async_client.post("/api/v1/diagnosis/quick-analyze")

        assert response.status_code == 422


class TestGetDiagnosis:
    """Tests for GET /api/v1/diagnosis/{diagnosis_id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_diagnosis_not_found_returns_404(
        self, async_client: AsyncClient, sample_dtc_codes
    ):
        """Test getting nonexistent diagnosis returns 404."""
        fake_id = uuid4()

        # Mock DiagnosisService to return None
        mock_service = AsyncMock()
        mock_service.get_diagnosis_by_id.return_value = None
        mock_service.__aenter__ = AsyncMock(return_value=mock_service)
        mock_service.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "app.api.v1.endpoints.diagnosis.DiagnosisService",
            return_value=mock_service,
        ):
            response = await async_client.get(f"/api/v1/diagnosis/{fake_id}")

            assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_diagnosis_invalid_uuid_returns_422(self, async_client: AsyncClient):
        """Test getting diagnosis with invalid UUID returns 422."""
        response = await async_client.get("/api/v1/diagnosis/not-a-uuid")

        assert response.status_code == 422


class TestDiagnosisHistory:
    """Tests for GET /api/v1/diagnosis/history/list endpoint."""

    @pytest.mark.asyncio
    async def test_history_requires_authentication(self, async_client: AsyncClient):
        """Test that history requires authentication."""
        response = await async_client.get("/api/v1/diagnosis/history/list")

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_history_returns_200_with_auth(
        self,
        async_client: AsyncClient,
        test_user,
        user_access_token: str,
    ):
        """Test that history returns 200 with authentication."""
        response = await async_client.get(
            "/api/v1/diagnosis/history/list",
            headers={"Authorization": f"Bearer {user_access_token}"},
        )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_history_returns_paginated_response(
        self,
        async_client: AsyncClient,
        test_user,
        user_access_token: str,
    ):
        """Test that history returns paginated response."""
        response = await async_client.get(
            "/api/v1/diagnosis/history/list",
            headers={"Authorization": f"Bearer {user_access_token}"},
        )

        assert response.status_code == 200
        data = response.json()

        assert "items" in data
        assert "total" in data
        assert "skip" in data
        assert "limit" in data
        assert "has_more" in data
        assert isinstance(data["items"], list)

    @pytest.mark.asyncio
    async def test_history_respects_skip_parameter(
        self,
        async_client: AsyncClient,
        test_user,
        user_access_token: str,
    ):
        """Test that history respects skip parameter."""
        response = await async_client.get(
            "/api/v1/diagnosis/history/list",
            headers={"Authorization": f"Bearer {user_access_token}"},
            params={"skip": 5},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["skip"] == 5

    @pytest.mark.asyncio
    async def test_history_respects_limit_parameter(
        self,
        async_client: AsyncClient,
        test_user,
        user_access_token: str,
    ):
        """Test that history respects limit parameter."""
        response = await async_client.get(
            "/api/v1/diagnosis/history/list",
            headers={"Authorization": f"Bearer {user_access_token}"},
            params={"limit": 5},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 5
        assert len(data["items"]) <= 5

    @pytest.mark.asyncio
    async def test_history_validates_limit_max(
        self,
        async_client: AsyncClient,
        test_user,
        user_access_token: str,
    ):
        """Test that history validates limit maximum."""
        response = await async_client.get(
            "/api/v1/diagnosis/history/list",
            headers={"Authorization": f"Bearer {user_access_token}"},
            params={"limit": 200},  # Over max of 100
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_history_filter_by_vehicle_make(
        self,
        async_client: AsyncClient,
        test_user,
        user_access_token: str,
        sample_diagnosis_session,
    ):
        """Test filtering history by vehicle make."""
        response = await async_client.get(
            "/api/v1/diagnosis/history/list",
            headers={"Authorization": f"Bearer {user_access_token}"},
            params={"vehicle_make": "Volkswagen"},
        )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_history_filter_by_vehicle_model(
        self,
        async_client: AsyncClient,
        test_user,
        user_access_token: str,
    ):
        """Test filtering history by vehicle model."""
        response = await async_client.get(
            "/api/v1/diagnosis/history/list",
            headers={"Authorization": f"Bearer {user_access_token}"},
            params={"vehicle_model": "Golf"},
        )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_history_filter_by_vehicle_year(
        self,
        async_client: AsyncClient,
        test_user,
        user_access_token: str,
    ):
        """Test filtering history by vehicle year."""
        response = await async_client.get(
            "/api/v1/diagnosis/history/list",
            headers={"Authorization": f"Bearer {user_access_token}"},
            params={"vehicle_year": 2018},
        )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_history_filter_by_dtc_code(
        self,
        async_client: AsyncClient,
        test_user,
        user_access_token: str,
    ):
        """Test filtering history by DTC code."""
        response = await async_client.get(
            "/api/v1/diagnosis/history/list",
            headers={"Authorization": f"Bearer {user_access_token}"},
            params={"dtc_code": "P0101"},
        )

        assert response.status_code == 200


class TestDeleteDiagnosis:
    """Tests for DELETE /api/v1/diagnosis/{diagnosis_id} endpoint."""

    @pytest.mark.asyncio
    async def test_delete_requires_authentication(self, async_client: AsyncClient):
        """Test that delete requires authentication."""
        fake_id = uuid4()
        response = await async_client.delete(f"/api/v1/diagnosis/{fake_id}")

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_delete_not_found_returns_404(
        self,
        async_client: AsyncClient,
        test_user,
        user_access_token: str,
    ):
        """Test deleting nonexistent diagnosis returns 404."""
        fake_id = uuid4()
        response = await async_client.delete(
            f"/api/v1/diagnosis/{fake_id}",
            headers={"Authorization": f"Bearer {user_access_token}"},
        )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_success_returns_200(
        self,
        async_client: AsyncClient,
        test_user,
        user_access_token: str,
        sample_diagnosis_session,
    ):
        """Test successful delete returns 200."""
        response = await async_client.delete(
            f"/api/v1/diagnosis/{sample_diagnosis_session.id}",
            headers={"Authorization": f"Bearer {user_access_token}"},
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "deleted_id" in data

    @pytest.mark.asyncio
    async def test_delete_invalid_uuid_returns_422(
        self,
        async_client: AsyncClient,
        test_user,
        user_access_token: str,
    ):
        """Test deleting with invalid UUID returns 422."""
        response = await async_client.delete(
            "/api/v1/diagnosis/not-a-uuid",
            headers={"Authorization": f"Bearer {user_access_token}"},
        )

        assert response.status_code == 422


class TestDiagnosisStats:
    """Tests for GET /api/v1/diagnosis/stats/summary endpoint."""

    @pytest.mark.asyncio
    async def test_stats_requires_authentication(self, async_client: AsyncClient):
        """Test that stats requires authentication."""
        response = await async_client.get("/api/v1/diagnosis/stats/summary")

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_stats_returns_200_with_auth(
        self,
        async_client: AsyncClient,
        test_user,
        user_access_token: str,
    ):
        """Test that stats returns 200 with authentication."""
        response = await async_client.get(
            "/api/v1/diagnosis/stats/summary",
            headers={"Authorization": f"Bearer {user_access_token}"},
        )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_stats_returns_expected_fields(
        self,
        async_client: AsyncClient,
        test_user,
        user_access_token: str,
    ):
        """Test that stats returns expected fields."""
        response = await async_client.get(
            "/api/v1/diagnosis/stats/summary",
            headers={"Authorization": f"Bearer {user_access_token}"},
        )

        assert response.status_code == 200
        data = response.json()

        assert "total_diagnoses" in data
        assert "avg_confidence" in data
        assert "most_diagnosed_vehicles" in data
        assert "most_common_dtcs" in data
        assert "diagnoses_by_month" in data

    @pytest.mark.asyncio
    async def test_stats_most_diagnosed_vehicles_is_list(
        self,
        async_client: AsyncClient,
        test_user,
        user_access_token: str,
    ):
        """Test that most_diagnosed_vehicles is a list."""
        response = await async_client.get(
            "/api/v1/diagnosis/stats/summary",
            headers={"Authorization": f"Bearer {user_access_token}"},
        )

        assert response.status_code == 200
        data = response.json()

        assert isinstance(data["most_diagnosed_vehicles"], list)

    @pytest.mark.asyncio
    async def test_stats_most_common_dtcs_is_list(
        self,
        async_client: AsyncClient,
        test_user,
        user_access_token: str,
    ):
        """Test that most_common_dtcs is a list."""
        response = await async_client.get(
            "/api/v1/diagnosis/stats/summary",
            headers={"Authorization": f"Bearer {user_access_token}"},
        )

        assert response.status_code == 200
        data = response.json()

        assert isinstance(data["most_common_dtcs"], list)


class TestDiagnosisResponseFormat:
    """Tests for diagnosis response format consistency."""

    @pytest.mark.asyncio
    async def test_history_item_format(
        self,
        async_client: AsyncClient,
        test_user,
        user_access_token: str,
        sample_diagnosis_session,
    ):
        """Test history item has correct format."""
        response = await async_client.get(
            "/api/v1/diagnosis/history/list",
            headers={"Authorization": f"Bearer {user_access_token}"},
        )

        assert response.status_code == 200
        data = response.json()

        if data["items"]:
            item = data["items"][0]
            assert "id" in item
            assert "vehicle_make" in item
            assert "vehicle_model" in item
            assert "vehicle_year" in item
            assert "dtc_codes" in item
            assert "confidence_score" in item
            assert "created_at" in item

    @pytest.mark.asyncio
    async def test_vehicle_diagnosis_count_format(
        self,
        async_client: AsyncClient,
        test_user,
        user_access_token: str,
    ):
        """Test vehicle diagnosis count has correct format."""
        response = await async_client.get(
            "/api/v1/diagnosis/stats/summary",
            headers={"Authorization": f"Bearer {user_access_token}"},
        )

        assert response.status_code == 200
        data = response.json()

        for vehicle in data["most_diagnosed_vehicles"]:
            assert "make" in vehicle
            assert "model" in vehicle
            assert "count" in vehicle

    @pytest.mark.asyncio
    async def test_dtc_frequency_format(
        self,
        async_client: AsyncClient,
        test_user,
        user_access_token: str,
    ):
        """Test DTC frequency has correct format."""
        response = await async_client.get(
            "/api/v1/diagnosis/stats/summary",
            headers={"Authorization": f"Bearer {user_access_token}"},
        )

        assert response.status_code == 200
        data = response.json()

        for dtc in data["most_common_dtcs"]:
            assert "code" in dtc
            assert "count" in dtc

    @pytest.mark.asyncio
    async def test_monthly_count_format(
        self,
        async_client: AsyncClient,
        test_user,
        user_access_token: str,
    ):
        """Test monthly count has correct format."""
        response = await async_client.get(
            "/api/v1/diagnosis/stats/summary",
            headers={"Authorization": f"Bearer {user_access_token}"},
        )

        assert response.status_code == 200
        data = response.json()

        for monthly in data["diagnoses_by_month"]:
            assert "month" in monthly
            assert "count" in monthly
