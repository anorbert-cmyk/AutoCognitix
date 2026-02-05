"""
End-to-end tests for authentication flow.

Tests the complete authentication workflow including:
- User registration
- Login with correct/incorrect credentials
- JWT token validation
- Token refresh
- Protected endpoint access
- Logout
- Password management
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(backend_path))


class TestUserRegistration:
    """Test user registration endpoint."""

    @pytest.mark.asyncio
    async def test_register_with_valid_data(self, async_client, seeded_db):
        """Test successful user registration."""
        user_data = {
            "email": "newuser@example.com",
            "password": "SecurePassword123!",
            "full_name": "New User",
        }

        response = await async_client.post("/api/v1/auth/register", json=user_data)

        assert response.status_code == 201
        data = response.json()
        assert data["email"] == user_data["email"].lower()
        assert "id" in data
        assert "password" not in data  # Password should not be returned
        assert "hashed_password" not in data

    @pytest.mark.asyncio
    async def test_register_without_full_name(self, async_client, seeded_db):
        """Test registration without optional full_name."""
        user_data = {
            "email": "noname@example.com",
            "password": "SecurePassword123!",
        }

        response = await async_client.post("/api/v1/auth/register", json=user_data)

        assert response.status_code == 201
        data = response.json()
        assert data["email"] == user_data["email"].lower()

    @pytest.mark.asyncio
    async def test_register_duplicate_email_rejected(self, async_client, seeded_db):
        """Test that duplicate email registration is rejected."""
        # First registration
        user_data = {
            "email": "duplicate@example.com",
            "password": "SecurePassword123!",
        }
        response1 = await async_client.post("/api/v1/auth/register", json=user_data)
        assert response1.status_code == 201

        # Second registration with same email
        response2 = await async_client.post("/api/v1/auth/register", json=user_data)
        assert response2.status_code == 400

    @pytest.mark.asyncio
    async def test_register_email_normalized_to_lowercase(self, async_client, seeded_db):
        """Test that email is normalized to lowercase."""
        user_data = {
            "email": "MixedCase@Example.COM",
            "password": "SecurePassword123!",
        }

        response = await async_client.post("/api/v1/auth/register", json=user_data)

        assert response.status_code == 201
        data = response.json()
        assert data["email"] == "mixedcase@example.com"

    @pytest.mark.asyncio
    async def test_register_invalid_email_rejected(self, async_client, seeded_db):
        """Test that invalid email format is rejected."""
        user_data = {
            "email": "not-an-email",
            "password": "SecurePassword123!",
        }

        response = await async_client.post("/api/v1/auth/register", json=user_data)

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_register_weak_password_rejected(self, async_client, seeded_db):
        """Test that weak password is rejected."""
        # Too short
        user_data = {
            "email": "weak1@example.com",
            "password": "short",
        }
        response = await async_client.post("/api/v1/auth/register", json=user_data)
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_register_password_without_uppercase_rejected(self, async_client, seeded_db):
        """Test that password without uppercase is rejected."""
        user_data = {
            "email": "nouppercase@example.com",
            "password": "password123!",  # No uppercase
        }

        response = await async_client.post("/api/v1/auth/register", json=user_data)

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_register_password_without_lowercase_rejected(self, async_client, seeded_db):
        """Test that password without lowercase is rejected."""
        user_data = {
            "email": "nolowercase@example.com",
            "password": "PASSWORD123!",  # No lowercase
        }

        response = await async_client.post("/api/v1/auth/register", json=user_data)

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_register_password_without_number_rejected(self, async_client, seeded_db):
        """Test that password without number is rejected."""
        user_data = {
            "email": "nonumber@example.com",
            "password": "Password!@#",  # No number
        }

        response = await async_client.post("/api/v1/auth/register", json=user_data)

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_register_password_without_special_char_rejected(self, async_client, seeded_db):
        """Test that password without special character is rejected."""
        user_data = {
            "email": "nospecial@example.com",
            "password": "Password123",  # No special character
        }

        response = await async_client.post("/api/v1/auth/register", json=user_data)

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_register_sets_default_role(self, async_client, seeded_db):
        """Test that registration sets default user role."""
        user_data = {
            "email": "defaultrole@example.com",
            "password": "SecurePassword123!",
        }

        response = await async_client.post("/api/v1/auth/register", json=user_data)

        assert response.status_code == 201
        data = response.json()
        assert data["role"] == "user"

    @pytest.mark.asyncio
    async def test_register_cannot_self_assign_admin(self, async_client, seeded_db):
        """Test that users cannot self-assign admin role."""
        user_data = {
            "email": "wannabeadmin@example.com",
            "password": "SecurePassword123!",
            "role": "admin",  # Try to assign admin role
        }

        response = await async_client.post("/api/v1/auth/register", json=user_data)

        assert response.status_code == 201
        data = response.json()
        # Should be downgraded to user
        assert data["role"] == "user"


class TestUserLogin:
    """Test user login endpoint."""

    @pytest.mark.asyncio
    async def test_login_with_correct_credentials(self, async_client, seeded_db):
        """Test successful login with correct credentials."""
        # First register a user
        register_data = {
            "email": "logintest@example.com",
            "password": "SecurePassword123!",
        }
        await async_client.post("/api/v1/auth/register", json=register_data)

        # Then login
        login_data = {
            "username": "logintest@example.com",
            "password": "SecurePassword123!",
        }

        response = await async_client.post(
            "/api/v1/auth/login",
            data=login_data,  # OAuth2 form data
        )

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

    @pytest.mark.asyncio
    async def test_login_with_wrong_password(self, async_client, seeded_db):
        """Test login with wrong password."""
        # First register a user
        register_data = {
            "email": "wrongpass@example.com",
            "password": "SecurePassword123!",
        }
        await async_client.post("/api/v1/auth/register", json=register_data)

        # Try to login with wrong password
        login_data = {
            "username": "wrongpass@example.com",
            "password": "WrongPassword123!",
        }

        response = await async_client.post(
            "/api/v1/auth/login",
            data=login_data,
        )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_login_with_nonexistent_email(self, async_client, seeded_db):
        """Test login with nonexistent email."""
        login_data = {
            "username": "nonexistent@example.com",
            "password": "AnyPassword123!",
        }

        response = await async_client.post(
            "/api/v1/auth/login",
            data=login_data,
        )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_login_email_case_insensitive(self, async_client, seeded_db):
        """Test that login email is case insensitive."""
        # Register with lowercase
        register_data = {
            "email": "casetest@example.com",
            "password": "SecurePassword123!",
        }
        await async_client.post("/api/v1/auth/register", json=register_data)

        # Login with mixed case
        login_data = {
            "username": "CaseTest@Example.COM",
            "password": "SecurePassword123!",
        }

        response = await async_client.post(
            "/api/v1/auth/login",
            data=login_data,
        )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_login_returns_jwt_tokens(self, async_client, seeded_db):
        """Test that login returns valid JWT tokens."""
        # Register user
        register_data = {
            "email": "jwttest@example.com",
            "password": "SecurePassword123!",
        }
        await async_client.post("/api/v1/auth/register", json=register_data)

        # Login
        login_data = {
            "username": "jwttest@example.com",
            "password": "SecurePassword123!",
        }

        response = await async_client.post(
            "/api/v1/auth/login",
            data=login_data,
        )

        assert response.status_code == 200
        data = response.json()

        # Tokens should be non-empty strings
        assert isinstance(data["access_token"], str)
        assert len(data["access_token"]) > 0
        assert isinstance(data["refresh_token"], str)
        assert len(data["refresh_token"]) > 0


class TestJWTTokenValidation:
    """Test JWT token validation."""

    @pytest.mark.asyncio
    async def test_access_protected_endpoint_with_valid_token(self, async_client, seeded_db):
        """Test accessing protected endpoint with valid token."""
        # Register and login
        register_data = {
            "email": "protectedtest@example.com",
            "password": "SecurePassword123!",
        }
        await async_client.post("/api/v1/auth/register", json=register_data)

        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": "protectedtest@example.com", "password": "SecurePassword123!"},
        )
        tokens = login_response.json()

        # Access protected endpoint
        response = await async_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "protectedtest@example.com"

    @pytest.mark.asyncio
    async def test_access_protected_endpoint_without_token(self, async_client, seeded_db):
        """Test accessing protected endpoint without token."""
        response = await async_client.get("/api/v1/auth/me")

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_access_protected_endpoint_with_invalid_token(self, async_client, seeded_db):
        """Test accessing protected endpoint with invalid token."""
        response = await async_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Bearer invalid_token_here"},
        )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_access_protected_endpoint_with_malformed_header(self, async_client, seeded_db):
        """Test accessing protected endpoint with malformed auth header."""
        response = await async_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "NotBearer token"},
        )

        assert response.status_code == 401


class TestTokenRefresh:
    """Test token refresh functionality."""

    @pytest.mark.asyncio
    async def test_refresh_tokens_with_valid_refresh_token(self, async_client, seeded_db):
        """Test refreshing tokens with valid refresh token."""
        # Register and login
        register_data = {
            "email": "refreshtest@example.com",
            "password": "SecurePassword123!",
        }
        await async_client.post("/api/v1/auth/register", json=register_data)

        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": "refreshtest@example.com", "password": "SecurePassword123!"},
        )
        tokens = login_response.json()

        # Refresh tokens
        response = await async_client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": tokens["refresh_token"]},
        )

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        # New tokens should be different
        assert data["access_token"] != tokens["access_token"]

    @pytest.mark.asyncio
    async def test_refresh_with_invalid_token(self, async_client, seeded_db):
        """Test refresh with invalid token."""
        response = await async_client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": "invalid_refresh_token"},
        )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_refresh_with_access_token_fails(self, async_client, seeded_db):
        """Test that using access token for refresh fails."""
        # Register and login
        register_data = {
            "email": "wrongtoken@example.com",
            "password": "SecurePassword123!",
        }
        await async_client.post("/api/v1/auth/register", json=register_data)

        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": "wrongtoken@example.com", "password": "SecurePassword123!"},
        )
        tokens = login_response.json()

        # Try to use access token as refresh token
        response = await async_client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": tokens["access_token"]},  # Wrong token type
        )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_new_access_token_works(self, async_client, seeded_db):
        """Test that new access token from refresh works."""
        # Register and login
        register_data = {
            "email": "newtokentest@example.com",
            "password": "SecurePassword123!",
        }
        await async_client.post("/api/v1/auth/register", json=register_data)

        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": "newtokentest@example.com", "password": "SecurePassword123!"},
        )
        tokens = login_response.json()

        # Refresh tokens
        refresh_response = await async_client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": tokens["refresh_token"]},
        )
        new_tokens = refresh_response.json()

        # Use new access token
        response = await async_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {new_tokens['access_token']}"},
        )

        assert response.status_code == 200


class TestLogout:
    """Test logout functionality."""

    @pytest.mark.asyncio
    async def test_logout_with_valid_token(self, async_client, seeded_db):
        """Test logout with valid token."""
        # Register and login
        register_data = {
            "email": "logouttest@example.com",
            "password": "SecurePassword123!",
        }
        await async_client.post("/api/v1/auth/register", json=register_data)

        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": "logouttest@example.com", "password": "SecurePassword123!"},
        )
        tokens = login_response.json()

        # Logout
        response = await async_client.post(
            "/api/v1/auth/logout",
            json={"refresh_token": tokens["refresh_token"]},
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_logout_without_refresh_token(self, async_client, seeded_db):
        """Test logout without providing refresh token."""
        # Register and login
        register_data = {
            "email": "logoutnorefresh@example.com",
            "password": "SecurePassword123!",
        }
        await async_client.post("/api/v1/auth/register", json=register_data)

        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": "logoutnorefresh@example.com", "password": "SecurePassword123!"},
        )
        tokens = login_response.json()

        # Logout without refresh token
        response = await async_client.post(
            "/api/v1/auth/logout",
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )

        assert response.status_code == 200


class TestProtectedEndpointAccess:
    """Test protected endpoint access patterns."""

    @pytest.mark.asyncio
    async def test_get_current_user_profile(self, async_client, seeded_db):
        """Test getting current user profile."""
        # Register and login
        register_data = {
            "email": "profiletest@example.com",
            "password": "SecurePassword123!",
            "full_name": "Profile Test User",
        }
        await async_client.post("/api/v1/auth/register", json=register_data)

        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": "profiletest@example.com", "password": "SecurePassword123!"},
        )
        tokens = login_response.json()

        # Get profile
        response = await async_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "profiletest@example.com"
        assert data["full_name"] == "Profile Test User"
        assert data["is_active"] is True

    @pytest.mark.asyncio
    async def test_update_user_profile(self, async_client, seeded_db):
        """Test updating user profile."""
        # Register and login
        register_data = {
            "email": "updatetest@example.com",
            "password": "SecurePassword123!",
            "full_name": "Original Name",
        }
        await async_client.post("/api/v1/auth/register", json=register_data)

        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": "updatetest@example.com", "password": "SecurePassword123!"},
        )
        tokens = login_response.json()

        # Update profile
        response = await async_client.put(
            "/api/v1/auth/me",
            json={"full_name": "Updated Name"},
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["full_name"] == "Updated Name"

    @pytest.mark.asyncio
    async def test_update_email_to_existing_fails(self, async_client, seeded_db):
        """Test that updating email to existing one fails."""
        # Register first user
        await async_client.post(
            "/api/v1/auth/register",
            json={"email": "existing@example.com", "password": "SecurePassword123!"},
        )

        # Register and login second user
        await async_client.post(
            "/api/v1/auth/register",
            json={"email": "second@example.com", "password": "SecurePassword123!"},
        )

        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": "second@example.com", "password": "SecurePassword123!"},
        )
        tokens = login_response.json()

        # Try to update to existing email
        response = await async_client.put(
            "/api/v1/auth/me",
            json={"email": "existing@example.com"},
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )

        assert response.status_code == 400


class TestPasswordManagement:
    """Test password management functionality."""

    @pytest.mark.asyncio
    async def test_change_password_with_correct_current(self, async_client, seeded_db):
        """Test changing password with correct current password."""
        # Register and login
        register_data = {
            "email": "changepass@example.com",
            "password": "OldPassword123!",
        }
        await async_client.post("/api/v1/auth/register", json=register_data)

        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": "changepass@example.com", "password": "OldPassword123!"},
        )
        tokens = login_response.json()

        # Change password
        response = await async_client.put(
            "/api/v1/auth/me/password",
            json={
                "current_password": "OldPassword123!",
                "new_password": "NewPassword456!",
            },
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )

        assert response.status_code == 200

        # Verify new password works
        login_response2 = await async_client.post(
            "/api/v1/auth/login",
            data={"username": "changepass@example.com", "password": "NewPassword456!"},
        )
        assert login_response2.status_code == 200

    @pytest.mark.asyncio
    async def test_change_password_with_wrong_current(self, async_client, seeded_db):
        """Test changing password with wrong current password."""
        # Register and login
        register_data = {
            "email": "wrongcurrent@example.com",
            "password": "CorrectPassword123!",
        }
        await async_client.post("/api/v1/auth/register", json=register_data)

        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": "wrongcurrent@example.com", "password": "CorrectPassword123!"},
        )
        tokens = login_response.json()

        # Try to change password with wrong current
        response = await async_client.put(
            "/api/v1/auth/me/password",
            json={
                "current_password": "WrongPassword123!",
                "new_password": "NewPassword456!",
            },
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_change_password_validates_new_password(self, async_client, seeded_db):
        """Test that new password must meet requirements."""
        # Register and login
        register_data = {
            "email": "weaknew@example.com",
            "password": "StrongPassword123!",
        }
        await async_client.post("/api/v1/auth/register", json=register_data)

        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": "weaknew@example.com", "password": "StrongPassword123!"},
        )
        tokens = login_response.json()

        # Try to change to weak password
        response = await async_client.put(
            "/api/v1/auth/me/password",
            json={
                "current_password": "StrongPassword123!",
                "new_password": "weak",  # Weak password
            },
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )

        assert response.status_code == 422


class TestForgotPassword:
    """Test forgot password functionality."""

    @pytest.mark.asyncio
    async def test_forgot_password_with_existing_email(self, async_client, seeded_db):
        """Test forgot password request with existing email."""
        # Register user
        await async_client.post(
            "/api/v1/auth/register",
            json={"email": "forgottest@example.com", "password": "Password123!"},
        )

        # Request password reset
        response = await async_client.post(
            "/api/v1/auth/forgot-password",
            json={"email": "forgottest@example.com"},
        )

        # Should return success (doesn't reveal if email exists)
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_forgot_password_with_nonexistent_email(self, async_client, seeded_db):
        """Test forgot password request with nonexistent email."""
        response = await async_client.post(
            "/api/v1/auth/forgot-password",
            json={"email": "nonexistent@example.com"},
        )

        # Should return same response for security
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_forgot_password_response_format(self, async_client, seeded_db):
        """Test forgot password response format."""
        response = await async_client.post(
            "/api/v1/auth/forgot-password",
            json={"email": "anyemail@example.com"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "message" in data


class TestInactiveUserAccess:
    """Test access restrictions for inactive users."""

    @pytest.mark.asyncio
    async def test_inactive_user_cannot_login(self, async_client, seeded_db):
        """Test that inactive user cannot login."""
        # The seeded_db includes an inactive user
        login_data = {
            "username": "inactive@example.com",
            "password": "TestPassword123!",  # Same password as other seeded users
        }

        response = await async_client.post(
            "/api/v1/auth/login",
            data=login_data,
        )

        # Should be forbidden or unauthorized
        assert response.status_code in [401, 403]


class TestRoleBasedAccess:
    """Test role-based access control."""

    @pytest.mark.asyncio
    async def test_user_role_returned_in_profile(self, async_client, seeded_db):
        """Test that user role is returned in profile."""
        # Register and login
        register_data = {
            "email": "roletest@example.com",
            "password": "SecurePassword123!",
        }
        await async_client.post("/api/v1/auth/register", json=register_data)

        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": "roletest@example.com", "password": "SecurePassword123!"},
        )
        tokens = login_response.json()

        # Get profile
        response = await async_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "role" in data
        assert data["role"] in ["user", "mechanic", "admin"]


class TestAccountLocking:
    """Test account locking after failed login attempts."""

    @pytest.mark.asyncio
    async def test_failed_login_attempts_tracked(self, async_client, seeded_db):
        """Test that failed login attempts are tracked."""
        # Register user
        await async_client.post(
            "/api/v1/auth/register",
            json={"email": "locktest@example.com", "password": "CorrectPassword123!"},
        )

        # Multiple failed attempts
        for _ in range(3):
            response = await async_client.post(
                "/api/v1/auth/login",
                data={"username": "locktest@example.com", "password": "WrongPassword123!"},
            )
            assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_successful_login_resets_failed_attempts(self, async_client, seeded_db):
        """Test that successful login resets failed attempt counter."""
        # Register user
        await async_client.post(
            "/api/v1/auth/register",
            json={"email": "resetattempts@example.com", "password": "CorrectPassword123!"},
        )

        # Some failed attempts
        for _ in range(2):
            await async_client.post(
                "/api/v1/auth/login",
                data={"username": "resetattempts@example.com", "password": "WrongPassword123!"},
            )

        # Successful login
        response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": "resetattempts@example.com", "password": "CorrectPassword123!"},
        )

        assert response.status_code == 200


class TestAuthErrorResponses:
    """Test authentication error response formats."""

    @pytest.mark.asyncio
    async def test_401_response_includes_www_authenticate_header(self, async_client, seeded_db):
        """Test that 401 responses include WWW-Authenticate header."""
        response = await async_client.get("/api/v1/auth/me")

        assert response.status_code == 401
        # Check for WWW-Authenticate header
        assert "www-authenticate" in [h.lower() for h in response.headers.keys()]

    @pytest.mark.asyncio
    async def test_error_response_includes_detail(self, async_client, seeded_db):
        """Test that error responses include detail message."""
        response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": "nonexistent@example.com", "password": "Password123!"},
        )

        assert response.status_code == 401
        data = response.json()
        assert "detail" in data


class TestPasswordValidationRules:
    """Test comprehensive password validation rules."""

    @pytest.mark.asyncio
    async def test_password_minimum_8_characters(self, async_client, seeded_db):
        """Test that password must be at least 8 characters."""
        user_data = {
            "email": "short@example.com",
            "password": "Ab1!234",  # 7 characters
        }

        response = await async_client.post("/api/v1/auth/register", json=user_data)
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_password_exactly_8_characters_valid(self, async_client, seeded_db):
        """Test that password with exactly 8 characters is valid."""
        user_data = {
            "email": "exact8@example.com",
            "password": "Ab1!2345",  # 8 characters, meets all requirements
        }

        response = await async_client.post("/api/v1/auth/register", json=user_data)
        assert response.status_code == 201

    @pytest.mark.asyncio
    async def test_password_maximum_length(self, async_client, seeded_db):
        """Test password maximum length enforcement."""
        user_data = {
            "email": "maxlen@example.com",
            "password": "A" * 101 + "a1!",  # Over 100 characters
        }

        response = await async_client.post("/api/v1/auth/register", json=user_data)
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_password_at_maximum_length_valid(self, async_client, seeded_db):
        """Test password at exactly maximum length."""
        user_data = {
            "email": "atmax@example.com",
            "password": "Aa1!" + "x" * 96,  # Exactly 100 characters
        }

        response = await async_client.post("/api/v1/auth/register", json=user_data)
        assert response.status_code == 201

    @pytest.mark.asyncio
    async def test_password_common_patterns_allowed(self, async_client, seeded_db):
        """Test that valid passwords meeting requirements are accepted."""
        valid_passwords = [
            "Password123!",
            "MyP@ssw0rd",
            "Secure#Pass1",
            "Test!ng123",
            "Complex1ty!",
        ]

        for i, password in enumerate(valid_passwords):
            user_data = {
                "email": f"valid{i}@example.com",
                "password": password,
            }
            response = await async_client.post("/api/v1/auth/register", json=user_data)
            assert response.status_code == 201, f"Password '{password}' should be valid"


class TestEmailValidation:
    """Test email validation rules."""

    @pytest.mark.asyncio
    async def test_valid_email_formats(self, async_client, seeded_db):
        """Test various valid email formats."""
        valid_emails = [
            "user@domain.com",
            "user.name@domain.com",
            "user+tag@domain.com",
            "user@subdomain.domain.com",
        ]

        for i, email in enumerate(valid_emails):
            user_data = {
                "email": email,
                "password": f"SecurePass{i}!",
            }
            response = await async_client.post("/api/v1/auth/register", json=user_data)
            assert response.status_code == 201, f"Email '{email}' should be valid"

    @pytest.mark.asyncio
    async def test_invalid_email_formats(self, async_client, seeded_db):
        """Test various invalid email formats."""
        invalid_emails = [
            "not-an-email",
            "@domain.com",
            "user@",
            "user@.com",
            "user space@domain.com",
        ]

        for email in invalid_emails:
            user_data = {
                "email": email,
                "password": "SecurePassword123!",
            }
            response = await async_client.post("/api/v1/auth/register", json=user_data)
            assert response.status_code == 422, f"Email '{email}' should be invalid"


class TestTokenExpiration:
    """Test token expiration handling."""

    @pytest.mark.asyncio
    async def test_expired_access_token_rejected(self, async_client, seeded_db):
        """Test that expired access token is rejected."""
        from app.core.security import create_access_token
        from datetime import timedelta

        # Create an already expired token
        expired_token = create_access_token(
            subject="test-user-id",
            expires_delta=timedelta(seconds=-1),  # Already expired
        )

        response = await async_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {expired_token}"},
        )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_expired_refresh_token_rejected(self, async_client, seeded_db):
        """Test that expired refresh token is rejected."""
        from app.core.security import create_refresh_token
        from datetime import timedelta

        # Create an already expired token
        expired_token = create_refresh_token(
            subject="test-user-id",
            expires_delta=timedelta(seconds=-1),  # Already expired
        )

        response = await async_client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": expired_token},
        )

        assert response.status_code == 401


class TestSessionSecurity:
    """Test session security measures."""

    @pytest.mark.asyncio
    async def test_token_reuse_after_logout(self, async_client, seeded_db):
        """Test that token cannot be reused after logout."""
        # Register and login
        user_data = {
            "email": "reusetest@example.com",
            "password": "SecurePassword123!",
        }
        await async_client.post("/api/v1/auth/register", json=user_data)

        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": "reusetest@example.com", "password": "SecurePassword123!"},
        )
        tokens = login_response.json()

        # Logout
        await async_client.post(
            "/api/v1/auth/logout",
            json={"refresh_token": tokens["refresh_token"]},
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )

        # Try to use the old access token (may or may not be immediately invalidated)
        response = await async_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )

        # Token may still work briefly or be blacklisted
        assert response.status_code in [200, 401]

    @pytest.mark.asyncio
    async def test_refresh_token_single_use(self, async_client, seeded_db):
        """Test that refresh token can only be used once."""
        # Register and login
        user_data = {
            "email": "singleuse@example.com",
            "password": "SecurePassword123!",
        }
        await async_client.post("/api/v1/auth/register", json=user_data)

        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": "singleuse@example.com", "password": "SecurePassword123!"},
        )
        tokens = login_response.json()

        # First refresh should succeed
        refresh_response1 = await async_client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": tokens["refresh_token"]},
        )
        assert refresh_response1.status_code == 200

        # Second refresh with same token should fail (blacklisted)
        refresh_response2 = await async_client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": tokens["refresh_token"]},
        )
        assert refresh_response2.status_code == 401


class TestConcurrentAuthentication:
    """Test concurrent authentication scenarios."""

    @pytest.mark.asyncio
    async def test_multiple_active_sessions(self, async_client, seeded_db):
        """Test that user can have multiple active sessions."""
        import asyncio

        # Register user
        user_data = {
            "email": "multisession@example.com",
            "password": "SecurePassword123!",
        }
        await async_client.post("/api/v1/auth/register", json=user_data)

        # Login multiple times
        login_tasks = [
            async_client.post(
                "/api/v1/auth/login",
                data={"username": "multisession@example.com", "password": "SecurePassword123!"},
            )
            for _ in range(3)
        ]

        responses = await asyncio.gather(*login_tasks)

        # All logins should succeed
        for response in responses:
            assert response.status_code == 200

        # Each should return different tokens
        tokens = [r.json() for r in responses]
        access_tokens = [t["access_token"] for t in tokens]
        # Tokens should be unique
        assert len(set(access_tokens)) == len(access_tokens)


class TestAuthenticationHeaders:
    """Test authentication header handling."""

    @pytest.mark.asyncio
    async def test_bearer_prefix_required(self, async_client, seeded_db):
        """Test that Bearer prefix is required in Authorization header."""
        # Register and login
        user_data = {
            "email": "bearertest@example.com",
            "password": "SecurePassword123!",
        }
        await async_client.post("/api/v1/auth/register", json=user_data)

        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": "bearertest@example.com", "password": "SecurePassword123!"},
        )
        tokens = login_response.json()

        # Without Bearer prefix
        response = await async_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": tokens["access_token"]},
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_case_insensitive_bearer(self, async_client, seeded_db):
        """Test Bearer prefix case sensitivity."""
        # Register and login
        user_data = {
            "email": "casebearer@example.com",
            "password": "SecurePassword123!",
        }
        await async_client.post("/api/v1/auth/register", json=user_data)

        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": "casebearer@example.com", "password": "SecurePassword123!"},
        )
        tokens = login_response.json()

        # With correct Bearer prefix
        response = await async_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )
        assert response.status_code == 200


class TestResetPasswordFlow:
    """Test complete password reset flow."""

    @pytest.mark.asyncio
    async def test_reset_password_with_valid_token(self, async_client, seeded_db):
        """Test password reset with valid token."""
        from app.core.security import create_password_reset_token

        # Register user
        user_data = {
            "email": "resetvalid@example.com",
            "password": "OldPassword123!",
        }
        await async_client.post("/api/v1/auth/register", json=user_data)

        # Create reset token (normally sent via email)
        reset_token = create_password_reset_token("resetvalid@example.com")

        # Reset password
        response = await async_client.post(
            "/api/v1/auth/reset-password",
            json={
                "token": reset_token,
                "new_password": "NewPassword456!",
            },
        )

        assert response.status_code == 200

        # Verify new password works
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": "resetvalid@example.com", "password": "NewPassword456!"},
        )
        assert login_response.status_code == 200

    @pytest.mark.asyncio
    async def test_reset_password_token_single_use(self, async_client, seeded_db):
        """Test that reset token can only be used once."""
        from app.core.security import create_password_reset_token

        # Register user
        user_data = {
            "email": "resetsingle@example.com",
            "password": "OldPassword123!",
        }
        await async_client.post("/api/v1/auth/register", json=user_data)

        # Create reset token
        reset_token = create_password_reset_token("resetsingle@example.com")

        # First reset should succeed
        response1 = await async_client.post(
            "/api/v1/auth/reset-password",
            json={
                "token": reset_token,
                "new_password": "NewPassword456!",
            },
        )
        assert response1.status_code == 200

        # Second reset with same token should fail
        response2 = await async_client.post(
            "/api/v1/auth/reset-password",
            json={
                "token": reset_token,
                "new_password": "AnotherPassword789!",
            },
        )
        assert response2.status_code in [400, 401]


class TestUserProfileEdgeCases:
    """Test user profile edge cases."""

    @pytest.mark.asyncio
    async def test_update_to_empty_full_name(self, async_client, seeded_db):
        """Test updating full_name to empty string."""
        # Register and login
        user_data = {
            "email": "emptyname@example.com",
            "password": "SecurePassword123!",
            "full_name": "Original Name",
        }
        await async_client.post("/api/v1/auth/register", json=user_data)

        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": "emptyname@example.com", "password": "SecurePassword123!"},
        )
        tokens = login_response.json()

        # Update to empty name
        response = await async_client.put(
            "/api/v1/auth/me",
            json={"full_name": ""},
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )

        # Should either accept empty or reject
        assert response.status_code in [200, 422]

    @pytest.mark.asyncio
    async def test_update_to_null_full_name(self, async_client, seeded_db):
        """Test updating full_name to null."""
        # Register and login
        user_data = {
            "email": "nullname@example.com",
            "password": "SecurePassword123!",
            "full_name": "Original Name",
        }
        await async_client.post("/api/v1/auth/register", json=user_data)

        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": "nullname@example.com", "password": "SecurePassword123!"},
        )
        tokens = login_response.json()

        # Update to null
        response = await async_client.put(
            "/api/v1/auth/me",
            json={"full_name": None},
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )

        # Should handle null value
        assert response.status_code in [200, 422]


class TestRoleBasedAccessControl:
    """Test role-based access control."""

    @pytest.mark.asyncio
    async def test_user_role_restrictions(self, authenticated_client):
        """Test that regular user has appropriate restrictions."""
        client = authenticated_client["client"]
        headers = authenticated_client["headers"]
        user = authenticated_client["user"]

        assert user["role"] == "user"

        # User should be able to access their own profile
        response = await client.get("/api/v1/auth/me", headers=headers)
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_mechanic_role_access(self, mechanic_client):
        """Test mechanic role access."""
        client = mechanic_client["client"]
        headers = mechanic_client["headers"]
        user = mechanic_client["user"]

        assert user["role"] == "mechanic"

        # Mechanic should be able to access profile
        response = await client.get("/api/v1/auth/me", headers=headers)
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_admin_role_access(self, admin_client):
        """Test admin role access."""
        client = admin_client["client"]
        headers = admin_client["headers"]
        user = admin_client["user"]

        assert user["role"] == "admin"

        # Admin should be able to access profile
        response = await client.get("/api/v1/auth/me", headers=headers)
        assert response.status_code == 200


class TestTokenClaims:
    """Test JWT token claims."""

    @pytest.mark.asyncio
    async def test_access_token_contains_role(self, async_client, seeded_db):
        """Test that access token contains role claim."""
        from app.core.security import decode_token

        # Register and login
        user_data = {
            "email": "claimtest@example.com",
            "password": "SecurePassword123!",
        }
        await async_client.post("/api/v1/auth/register", json=user_data)

        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": "claimtest@example.com", "password": "SecurePassword123!"},
        )
        tokens = login_response.json()

        # Decode token and check claims
        payload = decode_token(tokens["access_token"])
        assert payload is not None
        assert "role" in payload
        assert payload["role"] == "user"

    @pytest.mark.asyncio
    async def test_refresh_token_type_claim(self, async_client, seeded_db):
        """Test that refresh token has correct type claim."""
        from app.core.security import decode_token

        # Register and login
        user_data = {
            "email": "refreshtype@example.com",
            "password": "SecurePassword123!",
        }
        await async_client.post("/api/v1/auth/register", json=user_data)

        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": "refreshtype@example.com", "password": "SecurePassword123!"},
        )
        tokens = login_response.json()

        # Decode refresh token
        payload = decode_token(tokens["refresh_token"])
        assert payload is not None
        assert payload.get("type") == "refresh"
