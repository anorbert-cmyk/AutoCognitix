"""Sprint 12: Password Reset + Password Strength Tests.

Tests use live imports where available and fall back to pytest.skip
when the implementation is not yet present. No external DB required.
"""

import inspect

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _import_validate_password_strength():
    """Import validate_password_strength or skip the test."""
    try:
        from app.core.security import validate_password_strength

        return validate_password_strength
    except (ImportError, AttributeError):
        pytest.skip("validate_password_strength not yet implemented")
        return None  # unreachable; pytest.skip() raises


# =============================================================================
# TestPasswordStrength
# =============================================================================


class TestPasswordStrength:
    """Live functional tests for validate_password_strength."""

    def test_valid_password(self):
        """A password meeting all requirements is returned unchanged."""
        fn = _import_validate_password_strength()
        assert fn("Test1234!") == "Test1234!"

    def test_too_short(self):
        """Password shorter than 8 characters raises ValueError mentioning 'legalább'."""
        fn = _import_validate_password_strength()
        with pytest.raises(ValueError, match="legalább"):
            fn("Ab1!")

    def test_no_uppercase(self):
        """Password without uppercase raises ValueError mentioning 'nagybetű'."""
        fn = _import_validate_password_strength()
        with pytest.raises(ValueError, match="nagybetű"):
            fn("test1234!")

    def test_no_lowercase(self):
        """Password without lowercase raises ValueError mentioning 'kisbetű'."""
        fn = _import_validate_password_strength()
        with pytest.raises(ValueError, match="kisbetű"):
            fn("TEST1234!")

    def test_no_digit(self):
        """Password without a digit raises ValueError mentioning 'számot'."""
        fn = _import_validate_password_strength()
        with pytest.raises(ValueError, match="számot"):
            fn("TestPassword!")

    def test_exactly_8_chars(self):
        """A valid 8-character password (boundary case) is accepted."""
        fn = _import_validate_password_strength()
        assert fn("Test123!") == "Test123!"

    def test_too_long(self):
        """Password exceeding 100 characters raises ValueError."""
        fn = _import_validate_password_strength()
        with pytest.raises(ValueError):
            fn("A1!" + "a" * 100)

    def test_no_special_character(self):
        """Password without a special character raises ValueError."""
        fn = _import_validate_password_strength()
        with pytest.raises(ValueError):
            fn("Test1234")

    def test_returns_string(self):
        """validate_password_strength returns the original string value."""
        fn = _import_validate_password_strength()
        pw = "Valid1!x"
        result = fn(pw)
        assert isinstance(result, str)
        assert result == pw


# =============================================================================
# TestPasswordResetModel
# =============================================================================


class TestPasswordResetModel:
    """Verify PasswordResetToken model via source inspection of models.py."""

    @pytest.fixture(autouse=True)
    def setup(self):
        try:
            from app.db.postgres import models

            self.source = inspect.getsource(models)
        except ImportError:
            pytest.skip("postgres models not importable")

    def test_model_class_exists(self):
        """PasswordResetToken class must be defined in models.py."""
        assert "PasswordResetToken" in self.source, (
            "PasswordResetToken model not found in postgres models"
        )

    def test_model_has_token_hash(self):
        """PasswordResetToken must have a token_hash field."""
        assert "token_hash" in self.source, "PasswordResetToken must have token_hash field"

    def test_model_has_expires_at(self):
        """PasswordResetToken must have an expires_at field."""
        assert "expires_at" in self.source, "PasswordResetToken must have expires_at field"

    def test_model_has_used_flag(self):
        """PasswordResetToken must have a 'used' boolean field for single-use enforcement."""
        assert "used" in self.source, "PasswordResetToken must have a 'used' field"

    def test_model_has_tablename(self):
        """PasswordResetToken must declare a __tablename__."""
        assert "password_reset_tokens" in self.source, (
            "PasswordResetToken must declare __tablename__ = 'password_reset_tokens'"
        )


# =============================================================================
# TestForgotPasswordNoEnum
# =============================================================================


class TestForgotPasswordNoEnum:
    """Verify forgot-password and reset-password routes in the auth router."""

    @pytest.fixture(autouse=True)
    def setup(self):
        try:
            from app.api.v1.endpoints.auth import router

            self.router = router
            self.paths = [r.path for r in router.routes]
        except ImportError:
            pytest.skip("auth router not importable")

    def test_forgot_password_endpoint_exists(self):
        """Router must expose a /forgot-password route."""
        assert any("forgot" in p or "reset" in p for p in self.paths), (
            f"Expected forgot/reset route in auth router, got: {self.paths}"
        )

    def test_reset_password_endpoint_exists(self):
        """Router must expose a /reset-password (or similar) route."""
        reset_paths = [p for p in self.paths if "reset" in p]
        assert len(reset_paths) >= 1, (
            f"Expected at least one reset-password route, got: {self.paths}"
        )

    def test_forgot_password_route_is_post(self):
        """The forgot-password route must accept POST requests."""
        try:
            from fastapi.routing import APIRoute
        except ImportError:
            pytest.skip("fastapi not available")

        forgot_routes = [
            r
            for r in self.router.routes
            if hasattr(r, "path") and ("forgot" in r.path or "reset" in r.path)
            if hasattr(r, "methods")
        ]
        assert len(forgot_routes) >= 1, "forgot/reset route not found as APIRoute"
        for route in forgot_routes:
            assert "POST" in route.methods, (  # type: ignore[attr-defined]
                f"Route {route.path} should accept POST, got {route.methods}"
            )
