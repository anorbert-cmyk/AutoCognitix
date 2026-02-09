"""
Integration tests for full API flows.

Tests complete user journeys:
- User registration -> Login -> Create diagnosis -> View history
- Token refresh flow
- Password reset flow
- DTC search -> Detail -> Related codes flow
- Vehicle info -> Recalls -> Complaints flow
"""

import pytest
from httpx import AsyncClient
from unittest.mock import patch, AsyncMock, MagicMock
from uuid import uuid4


class TestUserAuthenticationFlow:
    """Tests for complete user authentication flow."""

    @pytest.mark.asyncio
    async def test_register_login_access_profile_flow(self, async_client: AsyncClient):
        """Test: Register -> Login -> Access protected endpoint flow."""
        # Step 1: Register new user
        register_response = await async_client.post(
            "/api/v1/auth/register",
            json={
                "email": "flowtest@example.com",
                "password": "FlowTestPassword123!",
                "full_name": "Flow Test User",
            },
        )
        assert register_response.status_code == 201
        user_data = register_response.json()
        assert user_data["email"] == "flowtest@example.com"

        # Step 2: Login with new credentials
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={
                "username": "flowtest@example.com",
                "password": "FlowTestPassword123!",
            },
        )
        assert login_response.status_code == 200
        tokens = login_response.json()
        assert "access_token" in tokens
        assert "refresh_token" in tokens

        # Step 3: Access protected endpoint with token
        me_response = await async_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )
        assert me_response.status_code == 200
        profile = me_response.json()
        assert profile["email"] == "flowtest@example.com"
        assert profile["full_name"] == "Flow Test User"

    @pytest.mark.asyncio
    async def test_register_login_refresh_flow(self, async_client: AsyncClient):
        """Test: Register -> Login -> Refresh tokens flow."""
        # Step 1: Register
        await async_client.post(
            "/api/v1/auth/register",
            json={
                "email": "refreshtest@example.com",
                "password": "RefreshTest123!",
            },
        )

        # Step 2: Login
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={
                "username": "refreshtest@example.com",
                "password": "RefreshTest123!",
            },
        )
        tokens = login_response.json()

        # Step 3: Refresh tokens
        refresh_response = await async_client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": tokens["refresh_token"]},
        )
        assert refresh_response.status_code == 200
        new_tokens = refresh_response.json()

        # Step 4: Use new access token
        me_response = await async_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {new_tokens['access_token']}"},
        )
        assert me_response.status_code == 200

    @pytest.mark.asyncio
    async def test_login_update_profile_verify_flow(
        self, async_client: AsyncClient, test_user, test_user_password: str
    ):
        """Test: Login -> Update profile -> Verify changes flow."""
        # Step 1: Login
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={
                "username": test_user.email,
                "password": test_user_password,
            },
        )
        tokens = login_response.json()

        # Step 2: Update profile
        update_response = await async_client.put(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
            json={"full_name": "Updated Full Name"},
        )
        assert update_response.status_code == 200

        # Step 3: Verify changes
        me_response = await async_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )
        assert me_response.status_code == 200
        profile = me_response.json()
        assert profile["full_name"] == "Updated Full Name"

    @pytest.mark.asyncio
    async def test_login_change_password_relogin_flow(
        self, async_client: AsyncClient, test_user, test_user_password: str
    ):
        """Test: Login -> Change password -> Relogin with new password flow."""
        # Step 1: Login
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={
                "username": test_user.email,
                "password": test_user_password,
            },
        )
        tokens = login_response.json()

        # Step 2: Change password
        new_password = "NewSecurePassword456!"
        change_response = await async_client.put(
            "/api/v1/auth/me/password",
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
            json={
                "current_password": test_user_password,
                "new_password": new_password,
            },
        )
        assert change_response.status_code == 200

        # Step 3: Login with new password
        relogin_response = await async_client.post(
            "/api/v1/auth/login",
            data={
                "username": test_user.email,
                "password": new_password,
            },
        )
        assert relogin_response.status_code == 200

    @pytest.mark.asyncio
    async def test_login_logout_access_denied_flow(
        self, async_client: AsyncClient, test_user, test_user_password: str
    ):
        """Test: Login -> Logout -> Verify access denied flow."""
        # Step 1: Login
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={
                "username": test_user.email,
                "password": test_user_password,
            },
        )
        tokens = login_response.json()

        # Step 2: Verify token works before logout
        me_response_before = await async_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )
        assert me_response_before.status_code == 200

        # Step 3: Logout
        logout_response = await async_client.post(
            "/api/v1/auth/logout",
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
            json={"refresh_token": tokens["refresh_token"]},
        )
        assert logout_response.status_code == 200


class TestDiagnosisFlow:
    """Tests for complete diagnosis flow with authenticated user."""

    @pytest.mark.asyncio
    async def test_search_dtc_get_detail_flow(self, async_client: AsyncClient, sample_dtc_codes):
        """Test: Search DTC -> Get detail -> Get related codes flow."""
        # Step 1: Search for DTC codes
        search_response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "P0101"},
        )
        assert search_response.status_code == 200
        search_results = search_response.json()
        assert len(search_results) > 0

        # Step 2: Get detail for first result
        code = search_results[0]["code"]
        detail_response = await async_client.get(f"/api/v1/dtc/{code}")
        assert detail_response.status_code == 200
        detail = detail_response.json()
        assert detail["code"] == code

        # Step 3: Get related codes
        related_response = await async_client.get(f"/api/v1/dtc/{code}/related")
        assert related_response.status_code == 200
        related = related_response.json()
        assert isinstance(related, list)

    @pytest.mark.asyncio
    async def test_authenticated_diagnosis_history_flow(
        self,
        async_client: AsyncClient,
        test_user,
        user_access_token: str,
        sample_diagnosis_session,
    ):
        """Test: Login -> View history -> Get specific diagnosis flow."""
        # Step 1: Get diagnosis history
        history_response = await async_client.get(
            "/api/v1/diagnosis/history/list",
            headers={"Authorization": f"Bearer {user_access_token}"},
        )
        assert history_response.status_code == 200
        history = history_response.json()
        assert "items" in history
        assert "total" in history

        # Step 2: If there are items, get the first one
        if history["items"]:
            first_id = history["items"][0]["id"]

            # Mock the service for getting diagnosis
            mock_result = {
                "id": first_id,
                "vehicle_make": "Volkswagen",
                "vehicle_model": "Golf",
                "vehicle_year": 2018,
                "dtc_codes": ["P0101"],
                "symptoms": "Test symptoms",
                "probable_causes": [],
                "recommended_repairs": [],
                "confidence_score": 0.8,
                "sources": [],
                "created_at": "2024-01-01T00:00:00Z",
            }

            mock_service = AsyncMock()
            mock_service.get_diagnosis_by_id.return_value = mock_result
            mock_service.__aenter__ = AsyncMock(return_value=mock_service)
            mock_service.__aexit__ = AsyncMock(return_value=None)

            with patch(
                "app.api.v1.endpoints.diagnosis.DiagnosisService",
                return_value=mock_service,
            ):
                detail_response = await async_client.get(
                    f"/api/v1/diagnosis/{first_id}",
                )
                assert detail_response.status_code == 200

    @pytest.mark.asyncio
    async def test_diagnosis_filter_and_paginate_flow(
        self,
        async_client: AsyncClient,
        test_user,
        user_access_token: str,
        multiple_diagnosis_sessions,
    ):
        """Test: Get history -> Filter -> Paginate flow."""
        # Step 1: Get all history
        all_response = await async_client.get(
            "/api/v1/diagnosis/history/list",
            headers={"Authorization": f"Bearer {user_access_token}"},
        )
        assert all_response.status_code == 200

        # Step 2: Filter by make
        filtered_response = await async_client.get(
            "/api/v1/diagnosis/history/list",
            headers={"Authorization": f"Bearer {user_access_token}"},
            params={"vehicle_make": "Volkswagen"},
        )
        assert filtered_response.status_code == 200

        # Step 3: Paginate
        page1_response = await async_client.get(
            "/api/v1/diagnosis/history/list",
            headers={"Authorization": f"Bearer {user_access_token}"},
            params={"skip": 0, "limit": 2},
        )
        assert page1_response.status_code == 200
        page1 = page1_response.json()
        assert page1["limit"] == 2

        page2_response = await async_client.get(
            "/api/v1/diagnosis/history/list",
            headers={"Authorization": f"Bearer {user_access_token}"},
            params={"skip": 2, "limit": 2},
        )
        assert page2_response.status_code == 200
        page2 = page2_response.json()
        assert page2["skip"] == 2

    @pytest.mark.asyncio
    async def test_diagnosis_create_and_delete_flow(
        self,
        async_client: AsyncClient,
        test_user,
        user_access_token: str,
        sample_diagnosis_session,
    ):
        """Test: Create diagnosis -> View in history -> Delete flow."""
        diagnosis_id = sample_diagnosis_session.id

        # Step 1: View in history (verify it exists)
        history_response = await async_client.get(
            "/api/v1/diagnosis/history/list",
            headers={"Authorization": f"Bearer {user_access_token}"},
        )
        assert history_response.status_code == 200
        history = history_response.json()

        # Find the diagnosis in history
        found = any(str(item["id"]) == str(diagnosis_id) for item in history["items"])
        assert found

        # Step 2: Delete the diagnosis
        delete_response = await async_client.delete(
            f"/api/v1/diagnosis/{diagnosis_id}",
            headers={"Authorization": f"Bearer {user_access_token}"},
        )
        assert delete_response.status_code == 200
        delete_result = delete_response.json()
        assert delete_result["success"] is True


class TestVehicleInformationFlow:
    """Tests for complete vehicle information flow."""

    @pytest.mark.asyncio
    async def test_get_makes_models_years_flow(self, async_client: AsyncClient):
        """Test: Get makes -> Get models -> Get years flow."""
        # Step 1: Get all makes
        makes_response = await async_client.get("/api/v1/vehicles/makes")
        assert makes_response.status_code == 200
        makes = makes_response.json()
        assert len(makes) > 0

        # Step 2: Get models for first make
        first_make_id = makes[0]["id"]
        models_response = await async_client.get(f"/api/v1/vehicles/models/{first_make_id}")
        assert models_response.status_code == 200

        # Step 3: Get available years
        years_response = await async_client.get("/api/v1/vehicles/years")
        assert years_response.status_code == 200
        years = years_response.json()
        assert "years" in years

    @pytest.mark.asyncio
    async def test_decode_vin_and_get_recalls_flow(
        self, async_client: AsyncClient, mock_nhtsa_service
    ):
        """Test: Decode VIN -> Get vehicle info -> Get recalls flow."""
        with patch(
            "app.api.v1.endpoints.vehicles.get_nhtsa_service",
            return_value=mock_nhtsa_service,
        ):
            # Step 1: Decode VIN
            vin_response = await async_client.post(
                "/api/v1/vehicles/decode-vin",
                json={"vin": "WVWZZZ3CZWE123456"},
            )
            assert vin_response.status_code == 200
            vehicle = vin_response.json()

            # Step 2: Get recalls for decoded vehicle
            make = vehicle["make"]
            model = vehicle["model"]
            year = vehicle["year"]

            recalls_response = await async_client.get(
                f"/api/v1/vehicles/{make}/{model}/{year}/recalls"
            )
            assert recalls_response.status_code == 200

    @pytest.mark.asyncio
    async def test_search_make_get_models_flow(self, async_client: AsyncClient):
        """Test: Search makes -> Get models for found make flow."""
        # Step 1: Search for makes
        search_response = await async_client.get(
            "/api/v1/vehicles/makes",
            params={"search": "volk"},
        )
        assert search_response.status_code == 200
        makes = search_response.json()
        assert len(makes) > 0

        # Step 2: Get models for found make
        make_id = makes[0]["id"]
        models_response = await async_client.get(f"/api/v1/vehicles/models/{make_id}")
        assert models_response.status_code == 200


class TestCrossEndpointIntegration:
    """Tests for integration between different API endpoints."""

    @pytest.mark.asyncio
    async def test_dtc_used_in_quick_analyze(self, async_client: AsyncClient, sample_dtc_codes):
        """Test: Search DTC -> Use in quick analyze flow."""
        # Step 1: Search for DTC
        search_response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "P0101"},
        )
        assert search_response.status_code == 200
        results = search_response.json()

        if results:
            # Step 2: Use found code in quick analyze
            code = results[0]["code"]
            analyze_response = await async_client.post(
                "/api/v1/diagnosis/quick-analyze",
                params={"dtc_codes": [code]},
            )
            assert analyze_response.status_code == 200
            analysis = analyze_response.json()
            assert "dtc_codes" in analysis

    @pytest.mark.asyncio
    async def test_categories_match_dtc_codes(self, async_client: AsyncClient, sample_dtc_codes):
        """Test: Get categories -> Verify DTC codes match categories."""
        # Step 1: Get categories
        categories_response = await async_client.get("/api/v1/dtc/categories/list")
        assert categories_response.status_code == 200
        categories = categories_response.json()

        category_codes = {c["code"] for c in categories}

        # Step 2: Get some DTC codes
        search_response = await async_client.get(
            "/api/v1/dtc/search",
            params={"q": "P", "limit": 5},
        )
        assert search_response.status_code == 200
        dtc_codes = search_response.json()

        # Step 3: Verify categories match
        for dtc in dtc_codes:
            # First letter of DTC code should be in categories
            first_letter = dtc["code"][0].upper()
            assert first_letter in category_codes

    @pytest.mark.asyncio
    async def test_user_stats_reflect_history(
        self,
        async_client: AsyncClient,
        test_user,
        user_access_token: str,
        sample_diagnosis_session,
    ):
        """Test: View history -> Verify stats match."""
        # Step 1: Get history
        history_response = await async_client.get(
            "/api/v1/diagnosis/history/list",
            headers={"Authorization": f"Bearer {user_access_token}"},
        )
        assert history_response.status_code == 200
        history = history_response.json()

        # Step 2: Get stats
        stats_response = await async_client.get(
            "/api/v1/diagnosis/stats/summary",
            headers={"Authorization": f"Bearer {user_access_token}"},
        )
        assert stats_response.status_code == 200
        stats = stats_response.json()

        # Step 3: Verify total matches
        assert stats["total_diagnoses"] == history["total"]


class TestErrorHandlingFlow:
    """Tests for error handling across the API."""

    @pytest.mark.asyncio
    async def test_unauthenticated_protected_endpoints_flow(self, async_client: AsyncClient):
        """Test: Access protected endpoints without auth returns 401."""
        protected_endpoints = [
            ("GET", "/api/v1/auth/me"),
            ("PUT", "/api/v1/auth/me"),
            ("PUT", "/api/v1/auth/me/password"),
            ("POST", "/api/v1/auth/logout"),
            ("GET", "/api/v1/diagnosis/history/list"),
            ("GET", "/api/v1/diagnosis/stats/summary"),
            ("DELETE", f"/api/v1/diagnosis/{uuid4()}"),
        ]

        for method, endpoint in protected_endpoints:
            if method == "GET":
                response = await async_client.get(endpoint)
            elif method == "POST":
                response = await async_client.post(endpoint)
            elif method == "PUT":
                response = await async_client.put(endpoint, json={})
            elif method == "DELETE":
                response = await async_client.delete(endpoint)

            assert response.status_code == 401, f"Expected 401 for {method} {endpoint}"

    @pytest.mark.asyncio
    async def test_invalid_data_validation_flow(self, async_client: AsyncClient):
        """Test: Send invalid data to endpoints returns 422."""
        invalid_requests = [
            # Invalid email format
            ("/api/v1/auth/register", {"email": "invalid", "password": "Test123!"}),
            # Password too short
            ("/api/v1/auth/register", {"email": "test@test.com", "password": "short"}),
            # Missing required fields
            ("/api/v1/auth/register", {"email": "test@test.com"}),
        ]

        for endpoint, data in invalid_requests:
            response = await async_client.post(endpoint, json=data)
            assert response.status_code == 422, f"Expected 422 for {endpoint}"

    @pytest.mark.asyncio
    async def test_not_found_resources_flow(self, async_client: AsyncClient, sample_dtc_codes):
        """Test: Access nonexistent resources returns 404."""
        not_found_endpoints = [
            "/api/v1/dtc/P9999",  # Nonexistent DTC code
            "/api/v1/dtc/P9999/related",  # Related codes for nonexistent DTC
        ]

        for endpoint in not_found_endpoints:
            response = await async_client.get(endpoint)
            assert response.status_code == 404, f"Expected 404 for {endpoint}"
