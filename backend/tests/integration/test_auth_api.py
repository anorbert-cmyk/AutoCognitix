"""
Integration tests for the authentication API endpoints.

Tests user registration, login, token refresh, and protected endpoints.
"""

import pytest
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(backend_path))


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
    async def test_login_returns_200(self, async_client, seeded_db, user_login_data):
        """Test that login returns 200 OK."""
        response = await async_client.post(
            "/api/v1/auth/login",
            data=user_login_data,  # OAuth2 form data
        )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_login_returns_tokens(self, async_client, seeded_db, user_login_data):
        """Test that login returns access and refresh tokens."""
        response = await async_client.post(
            "/api/v1/auth/login",
            data=user_login_data,
        )

        assert response.status_code == 200
        data = response.json()

        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

    @pytest.mark.asyncio
    async def test_login_tokens_are_jwt(self, async_client, seeded_db, user_login_data):
        """Test that login tokens are valid JWT format."""
        response = await async_client.post(
            "/api/v1/auth/login",
            data=user_login_data,
        )

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
        user_login_data,
    ):
        """Test that refresh returns 200 with valid refresh token."""
        # First, login to get tokens
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data=user_login_data,
        )
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
        user_login_data,
    ):
        """Test that refresh returns new access and refresh tokens."""
        # First, login to get tokens
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data=user_login_data,
        )
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
        user_login_data,
    ):
        """Test that refresh rejects access token (wrong type)."""
        # First, login to get tokens
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data=user_login_data,
        )
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
        user_login_data,
    ):
        """Test that /me returns 200 with valid access token."""
        # First, login to get tokens
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data=user_login_data,
        )
        tokens = login_response.json()

        # Then get current user
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
        user_login_data,
    ):
        """Test that /me returns user information."""
        # First, login to get tokens
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data=user_login_data,
        )
        tokens = login_response.json()

        # Then get current user
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
        user_login_data,
    ):
        """Test that /me rejects refresh token (wrong type)."""
        # First, login to get tokens
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data=user_login_data,
        )
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
        user_login_data,
    ):
        """Test that Bearer scheme is required."""
        # First, login to get tokens
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data=user_login_data,
        )
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
        user_login_data,
    ):
        """Test that access and refresh tokens are different."""
        response = await async_client.post(
            "/api/v1/auth/login",
            data=user_login_data,
        )

        data = response.json()
        assert data["access_token"] != data["refresh_token"]

    @pytest.mark.asyncio
    async def test_refreshed_tokens_are_different(
        self,
        async_client,
        seeded_db,
        user_login_data,
    ):
        """Test that refreshed tokens are different from original."""
        # Login
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data=user_login_data,
        )
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
        user_login_data,
    ):
        """Test that password is never returned in any response."""
        # Login
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data=user_login_data,
        )
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
        assert "detail" in data
