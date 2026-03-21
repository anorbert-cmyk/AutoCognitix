"""
Integration tests for the authentication API endpoints.

Tests user registration, login, token refresh, and protected endpoints.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

# Add backend to path
backend_path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(backend_path))


# =============================================================================
# Mock Redis-dependent security functions (no Redis in test environment)
# =============================================================================


@pytest.fixture(autouse=True)
def _mock_redis_security():
    """Mock token blacklist functions to avoid Redis dependency in tests.

    The security module uses a fail-closed policy: when Redis is unavailable,
    is_token_blacklisted returns True (= reject token). We mock both functions
    so tests don't depend on a running Redis instance.
    """
    with (
        patch(
            "app.core.security.is_token_blacklisted",
            new_callable=AsyncMock,
            return_value=False,
        ),
        patch(
            "app.core.security.blacklist_token",
            new_callable=AsyncMock,
            return_value=True,
        ),
    ):
        yield


# =============================================================================
# Helper: register + login flow for tests that need authenticated tokens
# =============================================================================

REGISTERED_EMAIL = "loginuser@example.com"
REGISTERED_PASSWORD = "SecurePassword123!"


async def _register_and_login(async_client):
    """Register a fresh user via the API, then login to get tokens."""
    # Register (the API hashes the password properly with bcrypt)
    await async_client.post(
        "/api/v1/auth/register",
        json={
            "email": REGISTERED_EMAIL,
            "password": REGISTERED_PASSWORD,
            "full_name": "Login Test User",
        },
    )

    # Login using OAuth2 form data
    login_response = await async_client.post(
        "/api/v1/auth/login",
        data={
            "username": REGISTERED_EMAIL,
            "password": REGISTERED_PASSWORD,
        },
    )
    return login_response


class TestUserRegistration:
    """Test POST /api/v1/auth/register endpoint."""

    @pytest.mark.asyncio
    async def test_register_returns_201(
        self,
        async_client,
        seeded_db,
        user_registration_data,
    ):
        """Test that registration returns 201 Created."""
        response = await async_client.post(
            "/api/v1/auth/register",
            json=user_registration_data,
        )

        assert response.status_code == 201

    @pytest.mark.asyncio
    async def test_register_returns_user_response(
        self,
        async_client,
        seeded_db,
        user_registration_data,
    ):
        """Test that registration returns user information."""
        response = await async_client.post(
            "/api/v1/auth/register",
            json=user_registration_data,
        )

        assert response.status_code == 201
        data = response.json()

        assert "id" in data
        assert data["email"] == user_registration_data["email"]
        assert data["is_active"]
        assert data["role"] == "user"

    @pytest.mark.asyncio
    async def test_register_does_not_return_password(
        self,
        async_client,
        seeded_db,
        user_registration_data,
    ):
        """Test that registration response does not include password."""
        response = await async_client.post(
            "/api/v1/auth/register",
            json=user_registration_data,
        )

        assert response.status_code == 201
        data = response.json()

        assert "password" not in data
        assert "hashed_password" not in data

    @pytest.mark.asyncio
    async def test_register_validates_email_format(self, async_client, seeded_db):
        """Test that registration validates email format."""
        response = await async_client.post(
            "/api/v1/auth/register",
            json={
                "email": "invalid-email",
                "password": "SecurePassword123!",
            },
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_register_validates_password_minimum_length(self, async_client, seeded_db):
        """Test that registration validates password minimum length."""
        response = await async_client.post(
            "/api/v1/auth/register",
            json={
                "email": "test@example.com",
                "password": "short",  # Too short
            },
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_register_validates_required_fields(self, async_client, seeded_db):
        """Test that registration requires email and password."""
        # Missing password
        response = await async_client.post(
            "/api/v1/auth/register",
            json={"email": "test@example.com"},
        )
        assert response.status_code == 422

        # Missing email
        response = await async_client.post(
            "/api/v1/auth/register",
            json={"password": "SecurePassword123!"},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_register_accepts_optional_full_name(
        self,
        async_client,
        seeded_db,
    ):
        """Test that registration accepts optional full_name."""
        response = await async_client.post(
            "/api/v1/auth/register",
            json={
                "email": "testuser@example.com",
                "password": "SecurePassword123!",
                "full_name": "Test User",
            },
        )

        assert response.status_code == 201


class TestUserLogin:
    """Test POST /api/v1/auth/login endpoint."""

    @pytest.mark.asyncio
    async def test_login_returns_200(self, async_client, seeded_db):
        """Test that login returns 200 OK."""
        login_response = await _register_and_login(async_client)
        assert login_response.status_code == 200

    @pytest.mark.asyncio
    async def test_login_returns_tokens(self, async_client, seeded_db):
        """Test that login returns access and refresh tokens."""
        response = await _register_and_login(async_client)

        assert response.status_code == 200
        data = response.json()

        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

    @pytest.mark.asyncio
    async def test_login_tokens_are_jwt(self, async_client, seeded_db):
        """Test that login tokens are valid JWT format."""
        response = await _register_and_login(async_client)

        assert response.status_code == 200
        data = response.json()

        # JWT tokens have 3 parts separated by dots
        access_parts = data["access_token"].split(".")
        refresh_parts = data["refresh_token"].split(".")

        assert len(access_parts) == 3
        assert len(refresh_parts) == 3

    @pytest.mark.asyncio
    async def test_login_requires_username(self, async_client, seeded_db):
        """Test that login requires username."""
        response = await async_client.post(
            "/api/v1/auth/login",
            data={"password": "TestPassword123!"},
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_login_requires_password(self, async_client, seeded_db):
        """Test that login requires password."""
        response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": "test@example.com"},
        )

        assert response.status_code == 422


class TestTokenRefresh:
    """Test POST /api/v1/auth/refresh endpoint."""

    @pytest.mark.asyncio
    async def test_refresh_returns_200_with_valid_token(
        self,
        async_client,
        seeded_db,
    ):
        """Test that refresh returns 200 with valid refresh token."""
        login_response = await _register_and_login(async_client)
        tokens = login_response.json()

        # Then refresh
        response = await async_client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": tokens["refresh_token"]},
        )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_refresh_returns_new_tokens(
        self,
        async_client,
        seeded_db,
    ):
        """Test that refresh returns new access and refresh tokens."""
        login_response = await _register_and_login(async_client)
        tokens = login_response.json()

        # Then refresh
        response = await async_client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": tokens["refresh_token"]},
        )

        assert response.status_code == 200
        data = response.json()

        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

    @pytest.mark.asyncio
    async def test_refresh_rejects_invalid_token(self, async_client, seeded_db):
        """Test that refresh rejects invalid token."""
        response = await async_client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": "invalid.token.here"},
        )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_refresh_rejects_access_token(
        self,
        async_client,
        seeded_db,
    ):
        """Test that refresh rejects access token (wrong type)."""
        login_response = await _register_and_login(async_client)
        tokens = login_response.json()

        # Try to use access token as refresh token
        response = await async_client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": tokens["access_token"]},
        )

        assert response.status_code == 401


class TestCurrentUser:
    """Test GET /api/v1/auth/me endpoint."""

    @pytest.mark.asyncio
    async def test_me_returns_200_with_valid_token(
        self,
        async_client,
        seeded_db,
    ):
        """Test that /me returns 200 with valid access token."""
        login_response = await _register_and_login(async_client)
        tokens = login_response.json()

        response = await async_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_me_returns_user_info(
        self,
        async_client,
        seeded_db,
    ):
        """Test that /me returns user information."""
        login_response = await _register_and_login(async_client)
        tokens = login_response.json()

        response = await async_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )

        assert response.status_code == 200
        data = response.json()

        assert "id" in data
        assert "email" in data
        assert "is_active" in data
        assert "role" in data

    @pytest.mark.asyncio
    async def test_me_rejects_no_token(self, async_client, seeded_db):
        """Test that /me rejects request without token."""
        response = await async_client.get("/api/v1/auth/me")

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_me_rejects_invalid_token(self, async_client, seeded_db):
        """Test that /me rejects invalid token."""
        response = await async_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Bearer invalid.token.here"},
        )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_me_rejects_refresh_token(
        self,
        async_client,
        seeded_db,
    ):
        """Test that /me rejects refresh token (wrong type)."""
        login_response = await _register_and_login(async_client)
        tokens = login_response.json()

        # Try to use refresh token as access token
        response = await async_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {tokens['refresh_token']}"},
        )

        assert response.status_code == 401


class TestAuthorizationHeader:
    """Test Authorization header handling."""

    @pytest.mark.asyncio
    async def test_bearer_scheme_required(
        self,
        async_client,
        seeded_db,
    ):
        """Test that Bearer scheme is required."""
        login_response = await _register_and_login(async_client)
        tokens = login_response.json()

        # Try without Bearer prefix
        response = await async_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": tokens["access_token"]},
        )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_empty_bearer_token_rejected(self, async_client, seeded_db):
        """Test that empty bearer token is rejected."""
        response = await async_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Bearer "},
        )

        assert response.status_code in [401, 403]


class TestTokenSecurity:
    """Test token security features."""

    @pytest.mark.asyncio
    async def test_tokens_are_different(
        self,
        async_client,
        seeded_db,
    ):
        """Test that access and refresh tokens are different."""
        response = await _register_and_login(async_client)

        data = response.json()
        assert data["access_token"] != data["refresh_token"]

    @pytest.mark.asyncio
    async def test_refreshed_tokens_are_different(
        self,
        async_client,
        seeded_db,
    ):
        """Test that refreshed tokens are different from original."""
        login_response = await _register_and_login(async_client)
        original_tokens = login_response.json()

        # Refresh
        refresh_response = await async_client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": original_tokens["refresh_token"]},
        )
        new_tokens = refresh_response.json()

        # New tokens should be different
        assert new_tokens["access_token"] != original_tokens["access_token"]
        assert new_tokens["refresh_token"] != original_tokens["refresh_token"]

    @pytest.mark.asyncio
    async def test_password_not_in_response(
        self,
        async_client,
        seeded_db,
    ):
        """Test that password is never returned in any response."""
        # Login
        login_response = await _register_and_login(async_client)
        assert "password" not in login_response.text.lower()

        # Get user
        tokens = login_response.json()
        me_response = await async_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )
        assert "password" not in me_response.text.lower()


class TestAuthErrorResponses:
    """Test authentication error responses."""

    @pytest.mark.asyncio
    async def test_401_response_format(self, async_client, seeded_db):
        """Test that 401 responses have proper format."""
        response = await async_client.get("/api/v1/auth/me")

        assert response.status_code == 401
        data = response.json()
        assert "detail" in data

    @pytest.mark.asyncio
    async def test_422_response_format(self, async_client, seeded_db):
        """Test that 422 responses have proper format."""
        response = await async_client.post(
            "/api/v1/auth/register",
            json={"email": "invalid"},
        )

        assert response.status_code == 422
        data = response.json()
        # Custom error handler wraps validation errors in {"error": {...}} format
        assert "error" in data
        error = data["error"]
        assert "code" in error
        assert "message" in error
        assert "details" in error
