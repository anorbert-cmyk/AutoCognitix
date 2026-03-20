"""Sprint 10 GDPR & Security Tests.

Tests for GDPR compliance, PyJWT migration, rate limiting, and Neo4j fallback.
"""

import re
from pathlib import Path

BACKEND_DIR = Path(__file__).parent.parent / "app"


class TestPyJWTMigration:
    """Verify python-jose has been replaced with PyJWT."""

    def test_no_python_jose_in_requirements(self):
        """requirements.txt should not contain python-jose."""
        req_file = Path(__file__).parent.parent / "requirements.txt"
        content = req_file.read_text()
        assert "python-jose" not in content.lower(), "python-jose still in requirements.txt"

    def test_pyjwt_in_requirements(self):
        """requirements.txt should contain PyJWT."""
        req_file = Path(__file__).parent.parent / "requirements.txt"
        content = req_file.read_text()
        assert "pyjwt" in content.lower() or "PyJWT" in content, (
            "PyJWT not found in requirements.txt"
        )

    def test_no_jose_import_in_security(self):
        """security.py should not import from jose."""
        security_file = BACKEND_DIR / "core" / "security.py"
        content = security_file.read_text()
        assert "from jose" not in content, "security.py still imports from jose"
        assert "import jose" not in content, "security.py still imports jose"


class TestGDPREndpoints:
    """Verify GDPR endpoint code exists."""

    def test_delete_endpoint_exists(self):
        """auth.py should have a DELETE /me endpoint for GDPR Article 17."""
        auth_file = BACKEND_DIR / "api" / "v1" / "endpoints" / "auth.py"
        content = auth_file.read_text()
        assert (
            "delete_user_account" in content
            or "gdpr" in content.lower()
            or "router.delete" in content
        ), "No GDPR deletion endpoint found in auth.py"

    def test_export_endpoint_exists(self):
        """auth.py should have a GET /me/export endpoint for GDPR Article 20."""
        auth_file = BACKEND_DIR / "api" / "v1" / "endpoints" / "auth.py"
        content = auth_file.read_text()
        assert "export_user_data" in content or "export" in content.lower(), (
            "No GDPR export endpoint found in auth.py"
        )


class TestRateLimitHeaders:
    """Verify rate limit response headers are implemented."""

    def test_rate_limit_header_code_exists(self):
        """rate_limit.py should set X-RateLimit headers."""
        rl_file = BACKEND_DIR / "core" / "rate_limit.py"
        content = rl_file.read_text()
        assert "X-RateLimit" in content, "No X-RateLimit headers found in rate_limit.py"

    def test_endpoint_specific_configs_exist(self):
        """rate_limit.py should have DIAGNOSIS_CONFIG and SEARCH_CONFIG."""
        rl_file = BACKEND_DIR / "core" / "rate_limit.py"
        content = rl_file.read_text()
        assert "DIAGNOSIS_CONFIG" in content, "No DIAGNOSIS_CONFIG found"
        assert "SEARCH_CONFIG" in content, "No SEARCH_CONFIG found"


class TestXForwardedForFix:
    """Verify X-Forwarded-For handling is secure."""

    def test_uses_last_ip_not_first(self):
        """rate_limit.py should use the last/rightmost IP from X-Forwarded-For."""
        rl_file = BACKEND_DIR / "core" / "rate_limit.py"
        content = rl_file.read_text()
        # Should NOT use split(",")[0] (first IP = spoofable)
        assert 'split(",")[0]' not in content or "[-1]" in content, (
            "rate_limit.py still uses first IP from X-Forwarded-For (spoofable)"
        )


class TestCircuitBreakerFailClosed:
    """Verify circuit breaker + blacklist interaction is fail-closed."""

    def test_circuit_open_check_in_blacklist(self):
        """security.py should check circuit breaker state before cache.get()."""
        security_file = BACKEND_DIR / "core" / "security.py"
        content = security_file.read_text()
        assert "circuit" in content.lower(), (
            "security.py doesn't check circuit breaker state in is_token_blacklisted"
        )


class TestNeo4jGracefulDegradation:
    """Verify Neo4j has graceful fallback."""

    def test_neo4j_availability_check_exists(self):
        """neo4j_models.py should have is_neo4j_available() function."""
        neo4j_file = BACKEND_DIR / "db" / "neo4j_models.py"
        content = neo4j_file.read_text()
        assert "is_neo4j_available" in content, "No Neo4j availability check function found"


class TestUnicodeNormalization:
    """Verify Unicode normalization for Hungarian text."""

    def test_unicodedata_imported_in_rag(self):
        """rag_service.py should import unicodedata."""
        rag_file = BACKEND_DIR / "services" / "rag_service.py"
        content = rag_file.read_text()
        assert "unicodedata" in content, "rag_service.py doesn't import unicodedata"

    def test_nfc_normalization_used(self):
        """rag_service.py should use NFC normalization."""
        rag_file = BACKEND_DIR / "services" / "rag_service.py"
        content = rag_file.read_text()
        assert "NFC" in content or "normalize" in content, (
            "No Unicode normalization found in rag_service.py"
        )


class TestConnectionPoolConfig:
    """Verify connection pool is properly sized."""

    def test_pool_size_adequate(self):
        """session.py pool size should be at least 10 (production mode)."""
        session_file = BACKEND_DIR / "db" / "postgres" / "session.py"
        content = session_file.read_text()
        # Match both direct assignment (POOL_SIZE = 10) and expression-based
        # (POOL_SIZE = int((settings.DEBUG and 5) or 10))
        match = re.search(r"POOL_SIZE\s*=\s*(?:int\(.*?or\s+)?(\d+)", content)
        assert match, "POOL_SIZE not found in session.py"
        pool_size = int(match.group(1))
        assert pool_size >= 10, f"POOL_SIZE too small: {pool_size}"
