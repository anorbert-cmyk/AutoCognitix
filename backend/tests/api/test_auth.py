"""
API tests for authentication endpoints.

Tests:
- POST /api/v1/auth/register - User registration
- POST /api/v1/auth/login - User login (OAuth2 password flow)
- POST /api/v1/auth/refresh - Token refresh
- POST /api/v1/auth/logout - User logout
- GET /api/v1/auth/me - Get current user
- PUT /api/v1/auth/me - Update user profile
- PUT /api/v1/auth/me/password - Change password
- POST /api/v1/auth/forgot-password - Request password reset
- POST /api/v1/auth/reset-password - Reset password with token
"""

import pytest
from httpx import AsyncClient


class TestUserRegistration:
    """Tests for POST /api/v1/auth/register endpoint."""

    @pytest.mark.asyncio
    async def test_register_success(self, async_client: AsyncClient, user_registration_data: dict):
        """Test successful user registration returns 201."""
        response = await async_client.post(
            "/api/v1/auth/register",
            json=user_registration_data,
        )

        assert response.status_code == 201
        data = response.json()

        assert "id" in data
        assert data["email"] == user_registration_data["email"]
        assert data["full_name"] == user_registration_data["full_name"]
        assert data["is_active"] is True
        assert data["role"] == "user"

    @pytest.mark.asyncio
    async def test_register_does_not_return_password(
        self, async_client: AsyncClient, user_registration_data: dict
    ):
        """Test that registration response never includes password."""
        response = await async_client.post(
            "/api/v1/auth/register",
            json=user_registration_data,
        )

        assert response.status_code == 201
        data = response.json()

        assert "password" not in data
        assert "hashed_password" not in data
        assert "password" not in response.text.lower()

    @pytest.mark.asyncio
    async def test_register_duplicate_email_returns_400(
        self, async_client: AsyncClient, test_user, test_user_password: str
    ):
        """Test that registering with existing email returns 400."""
        response = await async_client.post(
            "/api/v1/auth/register",
            json={
                "email": test_user.email,
                "password": "AnotherPassword123!",
                "full_name": "Duplicate User",
            },
        )

        assert response.status_code == 400
        assert (
            "email" in response.json()["detail"].lower()
            or "regisztrálva" in response.json()["detail"].lower()
        )

    @pytest.mark.asyncio
    async def test_register_invalid_email_returns_422(self, async_client: AsyncClient):
        """Test that invalid email format returns 422."""
        response = await async_client.post(
            "/api/v1/auth/register",
            json={
                "email": "not-an-email",
                "password": "SecurePassword123!",
            },
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_register_password_too_short_returns_422(self, async_client: AsyncClient):
        """Test that password under 8 characters returns 422."""
        response = await async_client.post(
            "/api/v1/auth/register",
            json={
                "email": "test@example.com",
                "password": "Short1!",
            },
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_register_password_without_uppercase_returns_422(self, async_client: AsyncClient):
        """Test that password without uppercase returns 422."""
        response = await async_client.post(
            "/api/v1/auth/register",
            json={
                "email": "test@example.com",
                "password": "password123!",  # No uppercase
            },
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_register_password_without_lowercase_returns_422(self, async_client: AsyncClient):
        """Test that password without lowercase returns 422."""
        response = await async_client.post(
            "/api/v1/auth/register",
            json={
                "email": "test@example.com",
                "password": "PASSWORD123!",  # No lowercase
            },
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_register_password_without_digit_returns_422(self, async_client: AsyncClient):
        """Test that password without digit returns 422."""
        response = await async_client.post(
            "/api/v1/auth/register",
            json={
                "email": "test@example.com",
                "password": "PasswordTest!",  # No digit
            },
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_register_password_without_special_char_returns_422(
        self, async_client: AsyncClient
    ):
        """Test that password without special character returns 422."""
        response = await async_client.post(
            "/api/v1/auth/register",
            json={
                "email": "test@example.com",
                "password": "Password123",  # No special char
            },
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_register_missing_email_returns_422(self, async_client: AsyncClient):
        """Test that missing email returns 422."""
        response = await async_client.post(
            "/api/v1/auth/register",
            json={"password": "SecurePassword123!"},
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_register_missing_password_returns_422(self, async_client: AsyncClient):
        """Test that missing password returns 422."""
        response = await async_client.post(
            "/api/v1/auth/register",
            json={"email": "test@example.com"},
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_register_prevents_admin_role_self_assignment(self, async_client: AsyncClient):
        """Test that users cannot register themselves as admin."""
        response = await async_client.post(
            "/api/v1/auth/register",
            json={
                "email": "sneaky@example.com",
                "password": "SecurePassword123!",
                "role": "admin",
            },
        )

        assert response.status_code == 201
        data = response.json()
        # Role should be downgraded to user
        assert data["role"] == "user"


class TestUserLogin:
    """Tests for POST /api/v1/auth/login endpoint."""

    @pytest.mark.asyncio
    async def test_login_success(
        self, async_client: AsyncClient, test_user, test_user_password: str
    ):
        """Test successful login returns 200 with tokens."""
        response = await async_client.post(
            "/api/v1/auth/login",
            data={
                "username": test_user.email,
                "password": test_user_password,
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

    @pytest.mark.asyncio
    async def test_login_returns_jwt_tokens(
        self, async_client: AsyncClient, test_user, test_user_password: str
    ):
        """Test that login returns valid JWT format tokens."""
        response = await async_client.post(
            "/api/v1/auth/login",
            data={
                "username": test_user.email,
                "password": test_user_password,
            },
        )

        assert response.status_code == 200
        data = response.json()

        # JWT tokens have 3 parts separated by dots
        access_parts = data["access_token"].split(".")
        refresh_parts = data["refresh_token"].split(".")

        assert len(access_parts) == 3, "Access token should have 3 parts"
        assert len(refresh_parts) == 3, "Refresh token should have 3 parts"

    @pytest.mark.asyncio
    async def test_login_wrong_email_returns_401(
        self, async_client: AsyncClient, test_user_password: str
    ):
        """Test that login with wrong email returns 401."""
        response = await async_client.post(
            "/api/v1/auth/login",
            data={
                "username": "nonexistent@example.com",
                "password": test_user_password,
            },
        )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_login_wrong_password_returns_401(self, async_client: AsyncClient, test_user):
        """Test that login with wrong password returns 401."""
        response = await async_client.post(
            "/api/v1/auth/login",
            data={
                "username": test_user.email,
                "password": "WrongPassword123!",
            },
        )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_login_inactive_user_returns_403(
        self, async_client: AsyncClient, inactive_user, test_user_password: str
    ):
        """Test that login with inactive user returns 403."""
        response = await async_client.post(
            "/api/v1/auth/login",
            data={
                "username": inactive_user.email,
                "password": test_user_password,
            },
        )

        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_login_missing_username_returns_422(self, async_client: AsyncClient):
        """Test that login without username returns 422."""
        response = await async_client.post(
            "/api/v1/auth/login",
            data={"password": "TestPassword123!"},
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_login_missing_password_returns_422(self, async_client: AsyncClient, test_user):
        """Test that login without password returns 422."""
        response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": test_user.email},
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_login_access_and_refresh_tokens_are_different(
        self, async_client: AsyncClient, test_user, test_user_password: str
    ):
        """Test that access and refresh tokens are different."""
        response = await async_client.post(
            "/api/v1/auth/login",
            data={
                "username": test_user.email,
                "password": test_user_password,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["access_token"] != data["refresh_token"]


class TestTokenRefresh:
    """Tests for POST /api/v1/auth/refresh endpoint."""

    @pytest.mark.asyncio
    async def test_refresh_success(
        self, async_client: AsyncClient, test_user, user_refresh_token: str
    ):
        """Test successful token refresh returns 200 with new tokens."""
        response = await async_client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": user_refresh_token},
        )

        assert response.status_code == 200
        data = response.json()

        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

    @pytest.mark.asyncio
    async def test_refresh_returns_different_tokens(
        self, async_client: AsyncClient, test_user, user_refresh_token: str
    ):
        """Test that refresh returns different tokens than original."""
        response = await async_client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": user_refresh_token},
        )

        assert response.status_code == 200
        data = response.json()

        # New tokens should be different
        assert data["refresh_token"] != user_refresh_token

    @pytest.mark.asyncio
    async def test_refresh_invalid_token_returns_401(self, async_client: AsyncClient):
        """Test that refresh with invalid token returns 401."""
        response = await async_client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": "invalid.token.here"},
        )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_refresh_with_access_token_returns_401(
        self, async_client: AsyncClient, user_access_token: str
    ):
        """Test that using access token for refresh returns 401."""
        response = await async_client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": user_access_token},
        )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_refresh_missing_token_returns_422(self, async_client: AsyncClient):
        """Test that refresh without token returns 422."""
        response = await async_client.post(
            "/api/v1/auth/refresh",
            json={},
        )

        assert response.status_code == 422


class TestLogout:
    """Tests for POST /api/v1/auth/logout endpoint."""

    @pytest.mark.asyncio
    async def test_logout_success(self, async_client: AsyncClient, user_access_token: str):
        """Test successful logout returns 200."""
        response = await async_client.post(
            "/api/v1/auth/logout",
            headers={"Authorization": f"Bearer {user_access_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "message" in data

    @pytest.mark.asyncio
    async def test_logout_with_refresh_token(
        self,
        async_client: AsyncClient,
        user_access_token: str,
        user_refresh_token: str,
    ):
        """Test logout with refresh token in body."""
        response = await async_client.post(
            "/api/v1/auth/logout",
            headers={"Authorization": f"Bearer {user_access_token}"},
            json={"refresh_token": user_refresh_token},
        )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_logout_without_auth_returns_401(self, async_client: AsyncClient):
        """Test that logout without authentication returns 401."""
        response = await async_client.post("/api/v1/auth/logout")

        assert response.status_code == 401


class TestCurrentUser:
    """Tests for GET /api/v1/auth/me endpoint."""

    @pytest.mark.asyncio
    async def test_me_success(self, async_client: AsyncClient, test_user, user_access_token: str):
        """Test get current user returns 200 with user data."""
        response = await async_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {user_access_token}"},
        )

        assert response.status_code == 200
        data = response.json()

        assert data["id"] == str(test_user.id)
        assert data["email"] == test_user.email
        assert data["full_name"] == test_user.full_name
        assert data["is_active"] == test_user.is_active
        assert data["role"] == test_user.role

    @pytest.mark.asyncio
    async def test_me_does_not_return_password(
        self, async_client: AsyncClient, user_access_token: str
    ):
        """Test that /me never returns password."""
        response = await async_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {user_access_token}"},
        )

        assert response.status_code == 200
        assert "password" not in response.text.lower()
        assert "hashed" not in response.text.lower()

    @pytest.mark.asyncio
    async def test_me_without_auth_returns_401(self, async_client: AsyncClient):
        """Test that /me without authentication returns 401."""
        response = await async_client.get("/api/v1/auth/me")

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_me_with_invalid_token_returns_401(self, async_client: AsyncClient):
        """Test that /me with invalid token returns 401."""
        response = await async_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Bearer invalid.token.here"},
        )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_me_with_refresh_token_returns_401(
        self, async_client: AsyncClient, user_refresh_token: str
    ):
        """Test that /me with refresh token (wrong type) returns 401."""
        response = await async_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {user_refresh_token}"},
        )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_me_bearer_scheme_required(
        self, async_client: AsyncClient, user_access_token: str
    ):
        """Test that Bearer scheme is required."""
        response = await async_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": user_access_token},  # Missing "Bearer "
        )

        assert response.status_code == 401


class TestUpdateProfile:
    """Tests for PUT /api/v1/auth/me endpoint."""

    @pytest.mark.asyncio
    async def test_update_profile_full_name(
        self, async_client: AsyncClient, test_user, user_access_token: str
    ):
        """Test updating profile full_name."""
        response = await async_client.put(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {user_access_token}"},
            json={"full_name": "Updated Name"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["full_name"] == "Updated Name"

    @pytest.mark.asyncio
    async def test_update_profile_email(
        self, async_client: AsyncClient, test_user, user_access_token: str
    ):
        """Test updating profile email."""
        response = await async_client.put(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {user_access_token}"},
            json={"email": "newemail@example.com"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "newemail@example.com"

    @pytest.mark.asyncio
    async def test_update_profile_without_auth_returns_401(self, async_client: AsyncClient):
        """Test that updating profile without auth returns 401."""
        response = await async_client.put(
            "/api/v1/auth/me",
            json={"full_name": "New Name"},
        )

        assert response.status_code == 401


class TestChangePassword:
    """Tests for PUT /api/v1/auth/me/password endpoint."""

    @pytest.mark.asyncio
    async def test_change_password_success(
        self, async_client: AsyncClient, test_user, user_access_token: str, test_user_password: str
    ):
        """Test successful password change."""
        response = await async_client.put(
            "/api/v1/auth/me/password",
            headers={"Authorization": f"Bearer {user_access_token}"},
            json={
                "current_password": test_user_password,
                "new_password": "NewSecurePassword123!",
            },
        )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_change_password_wrong_current_password_returns_400(
        self, async_client: AsyncClient, test_user, user_access_token: str
    ):
        """Test that wrong current password returns 400."""
        response = await async_client.put(
            "/api/v1/auth/me/password",
            headers={"Authorization": f"Bearer {user_access_token}"},
            json={
                "current_password": "WrongPassword123!",
                "new_password": "NewSecurePassword123!",
            },
        )

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_change_password_weak_new_password_returns_422(
        self, async_client: AsyncClient, test_user, user_access_token: str, test_user_password: str
    ):
        """Test that weak new password returns 422."""
        response = await async_client.put(
            "/api/v1/auth/me/password",
            headers={"Authorization": f"Bearer {user_access_token}"},
            json={
                "current_password": test_user_password,
                "new_password": "weak",  # Too short, no uppercase, etc.
            },
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_change_password_without_auth_returns_401(self, async_client: AsyncClient):
        """Test that changing password without auth returns 401."""
        response = await async_client.put(
            "/api/v1/auth/me/password",
            json={
                "current_password": "Current123!",
                "new_password": "NewPassword123!",
            },
        )

        assert response.status_code == 401


class TestForgotPassword:
    """Tests for POST /api/v1/auth/forgot-password endpoint."""

    @pytest.mark.asyncio
    async def test_forgot_password_existing_email(self, async_client: AsyncClient, test_user):
        """Test forgot password with existing email returns 200."""
        response = await async_client.post(
            "/api/v1/auth/forgot-password",
            json={"email": test_user.email},
        )

        assert response.status_code == 200
        data = response.json()
        assert "message" in data

    @pytest.mark.asyncio
    async def test_forgot_password_nonexistent_email_returns_200(self, async_client: AsyncClient):
        """Test forgot password with nonexistent email returns 200 (security)."""
        # For security, should not reveal if email exists
        response = await async_client.post(
            "/api/v1/auth/forgot-password",
            json={"email": "nonexistent@example.com"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "message" in data

    @pytest.mark.asyncio
    async def test_forgot_password_invalid_email_returns_422(self, async_client: AsyncClient):
        """Test forgot password with invalid email returns 422."""
        response = await async_client.post(
            "/api/v1/auth/forgot-password",
            json={"email": "not-an-email"},
        )

        assert response.status_code == 422


class TestResetPassword:
    """Tests for POST /api/v1/auth/reset-password endpoint."""

    @pytest.mark.asyncio
    async def test_reset_password_invalid_token_returns_400(self, async_client: AsyncClient):
        """Test reset password with invalid token returns 400."""
        response = await async_client.post(
            "/api/v1/auth/reset-password",
            json={
                "token": "invalid.token.here",
                "new_password": "NewSecurePassword123!",
            },
        )

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_reset_password_weak_password_returns_422(self, async_client: AsyncClient):
        """Test reset password with weak password returns 422."""
        response = await async_client.post(
            "/api/v1/auth/reset-password",
            json={
                "token": "some.token.here",
                "new_password": "weak",
            },
        )

        assert response.status_code == 422


class TestAuthErrorResponses:
    """Tests for authentication error response formats."""

    @pytest.mark.asyncio
    async def test_401_response_has_detail(self, async_client: AsyncClient):
        """Test that 401 responses have 'detail' field."""
        response = await async_client.get("/api/v1/auth/me")

        assert response.status_code == 401
        data = response.json()
        assert "detail" in data

    @pytest.mark.asyncio
    async def test_422_response_has_detail(self, async_client: AsyncClient):
        """Test that 422 responses have 'detail' field."""
        response = await async_client.post(
            "/api/v1/auth/register",
            json={"email": "invalid"},
        )

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data


class TestAuthSecurityHeaders:
    """Tests for authentication security header handling."""

    @pytest.mark.asyncio
    async def test_empty_bearer_token_rejected(self, async_client: AsyncClient):
        """Test that empty bearer token is rejected."""
        response = await async_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Bearer "},
        )

        assert response.status_code in [401, 403]

    @pytest.mark.asyncio
    async def test_malformed_auth_header_rejected(self, async_client: AsyncClient):
        """Test that malformed Authorization header is rejected."""
        response = await async_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "NotBearer token"},
        )

        assert response.status_code == 401
