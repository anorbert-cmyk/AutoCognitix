"""Integration tests for the header-only HMAC-signed CSRF middleware.

These tests mount the REAL ``CSRFMiddleware`` (and the real
``generate_csrf_token`` / ``verify_csrf_token``) on a minimal app, using the
exact ``exclude_paths`` configured in ``app.main``. This is the coverage that
was previously missing: the app-level test clients build a bare ``FastAPI()``
without the middleware, so a broken CSRF flow (login 403ing, or writes never
validating) passed CI unnoticed.

The middleware now validates a signed token from the ``X-CSRF-Token`` header
only — no cookie — because production is cross-site and a ``SameSite`` cookie
is never sent cross-origin.
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from app.core.csrf import CSRFMiddleware
from app.core.security import generate_csrf_token, verify_csrf_token

# Must mirror app.main.create_application()'s CSRFMiddleware exclude_paths.
# NOTE: /api/v1/auth/refresh is intentionally NOT excluded — it rotates the
# refresh cookie and must stay CSRF-protected (see test_refresh_is_protected).
EXCLUDE_PATHS = [
    "/health",
    "/metrics",
    "/api/v1/docs",
    "/api/v1/openapi.json",
    "/api/v1/redoc",
    "/api/v1/auth/login",
    "/api/v1/auth/register",
    "/api/v1/auth/forgot-password",
    "/api/v1/auth/reset-password",
]


def _build_client() -> TestClient:
    app = FastAPI()

    @app.post("/api/v1/auth/login")
    async def login():  # excluded bootstrap endpoint
        return JSONResponse({"csrf_token": generate_csrf_token()})

    @app.post("/api/v1/garage/vehicles")
    async def create_vehicle():  # protected state-changing endpoint
        return JSONResponse({"ok": True})

    @app.get("/api/v1/dtc/search")
    async def search():  # safe method
        return JSONResponse({"ok": True})

    @app.post("/api/v1/auth/refresh")
    async def refresh():  # protected: rotates the refresh cookie
        return JSONResponse({"ok": True})

    @app.get("/api/v1/auth/csrf-token")
    async def csrf_token():  # safe bootstrap endpoint
        return JSONResponse({"csrf_token": generate_csrf_token()})

    app.add_middleware(CSRFMiddleware, exclude_paths=EXCLUDE_PATHS)
    return TestClient(app)


class TestCSRFMiddleware:
    def test_login_is_excluded_without_token(self):
        """A fresh login POST has no CSRF token yet; it must NOT be blocked."""
        client = _build_client()
        resp = client.post("/api/v1/auth/login", json={"username": "u", "password": "p"})
        assert resp.status_code == 200
        # login mints a signed token the client will echo on later writes
        assert "." in resp.json()["csrf_token"]

    def test_protected_write_without_token_is_forbidden(self):
        """A state-changing request with no X-CSRF-Token is rejected (fail closed)."""
        client = _build_client()
        resp = client.post("/api/v1/garage/vehicles", json={"vin": "X"})
        assert resp.status_code == 403
        assert resp.json()["code"] == "csrf_token_invalid"

    def test_protected_write_with_valid_token_succeeds(self):
        """A valid signed token in the header passes validation."""
        client = _build_client()
        token = generate_csrf_token()
        resp = client.post(
            "/api/v1/garage/vehicles",
            json={"vin": "X"},
            headers={"X-CSRF-Token": token},
        )
        assert resp.status_code == 200

    def test_protected_write_with_tampered_token_is_forbidden(self):
        """A token with a forged signature is rejected."""
        client = _build_client()
        token = generate_csrf_token()
        nonce, _, mac = token.partition(".")
        forged = f"{nonce}.{'0' * len(mac)}"
        resp = client.post(
            "/api/v1/garage/vehicles",
            json={"vin": "X"},
            headers={"X-CSRF-Token": forged},
        )
        assert resp.status_code == 403
        assert resp.json()["code"] == "csrf_token_invalid"

    def test_protected_write_with_random_header_is_forbidden(self):
        """An unsigned random string (old presence-only bypass) is now rejected."""
        client = _build_client()
        resp = client.post(
            "/api/v1/garage/vehicles",
            json={"vin": "X"},
            headers={"X-CSRF-Token": "just-a-random-32-character-string!!"},
        )
        assert resp.status_code == 403

    def test_safe_get_passes_and_sets_no_cookie(self):
        """Safe methods pass through and no csrf_token cookie is minted anymore."""
        client = _build_client()
        resp = client.get("/api/v1/dtc/search")
        assert resp.status_code == 200
        assert "csrf_token" not in resp.cookies

    def test_refresh_is_protected_without_token(self):
        """/auth/refresh rotates the refresh cookie, so it must stay CSRF-protected
        (a CSRF-less cross-origin POST could force a logout under SameSite=None)."""
        client = _build_client()
        resp = client.post("/api/v1/auth/refresh")
        assert resp.status_code == 403
        assert resp.json()["code"] == "csrf_token_invalid"

    def test_refresh_with_valid_token_succeeds(self):
        """/auth/refresh works when the client presents its signed CSRF token."""
        client = _build_client()
        resp = client.post("/api/v1/auth/refresh", headers={"X-CSRF-Token": generate_csrf_token()})
        assert resp.status_code == 200

    def test_csrf_token_bootstrap_is_safe_and_returns_token(self):
        """The reload bootstrap endpoint is a safe GET that hands out a signed token."""
        client = _build_client()
        resp = client.get("/api/v1/auth/csrf-token")
        assert resp.status_code == 200
        assert verify_csrf_token(resp.json()["csrf_token"]) is True


class TestCSRFTokenHelpers:
    def test_round_trip(self):
        assert verify_csrf_token(generate_csrf_token()) is True

    def test_rejects_missing_and_malformed(self):
        assert verify_csrf_token(None) is False
        assert verify_csrf_token("") is False
        assert verify_csrf_token("no-dot-here") is False
        assert verify_csrf_token("nonce.") is False
        assert verify_csrf_token(".mac") is False

    def test_rejects_forged_signature(self):
        token = generate_csrf_token()
        nonce, _, mac = token.partition(".")
        assert verify_csrf_token(f"{nonce}.{'0' * len(mac)}") is False

    def test_rejects_non_ascii_without_raising(self):
        """A non-ASCII token must return False, not raise. Starlette decodes
        headers as latin-1, so a raw 0x80-0xFF byte reaches verify as a
        non-ASCII str; hmac.compare_digest would TypeError (-> 500) without the
        isascii() guard. This is the server-side unit check (httpx's TestClient
        refuses to send such a header, so it can't be exercised end-to-end)."""
        assert verify_csrf_token("abc.\xe9") is False
        assert verify_csrf_token("\xe9.abc") is False

    def test_tokens_are_unique(self):
        assert generate_csrf_token() != generate_csrf_token()
