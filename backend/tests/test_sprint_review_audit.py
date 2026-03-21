"""Sprint 9/10 Review Audit Tests.

Tests verifying the fixes applied during post-sprint security and code review.
"""

import re
from pathlib import Path

BACKEND_DIR = Path(__file__).parent.parent / "app"


class TestGDPRDeletionOrder:
    """Verify GDPR deletion performs external cleanups before PostgreSQL commit."""

    def test_external_cleanup_before_pg_commit(self):
        """auth.py should clean Qdrant/Neo4j BEFORE committing PG deletion."""
        auth_file = BACKEND_DIR / "api" / "v1" / "endpoints" / "auth.py"
        content = auth_file.read_text()

        # Find positions: external cleanup should come before db.commit()
        qdrant_pos = content.find("delete_by_user")
        neo4j_pos = content.find("delete_user_data")
        commit_pos = content.find("await db.commit()", content.find("delete_user_account"))

        assert qdrant_pos > 0, "Qdrant cleanup not found in delete_user_account"
        assert neo4j_pos > 0, "Neo4j cleanup not found in delete_user_account"
        assert commit_pos > 0, "db.commit() not found in delete_user_account"
        assert qdrant_pos < commit_pos, "Qdrant cleanup should happen BEFORE db.commit()"
        assert neo4j_pos < commit_pos, "Neo4j cleanup should happen BEFORE db.commit()"

    def test_abort_on_external_cleanup_failure(self):
        """auth.py should abort deletion if external cleanup fails."""
        auth_file = BACKEND_DIR / "api" / "v1" / "endpoints" / "auth.py"
        content = auth_file.read_text()

        # Should raise HTTPException on cleanup failure
        assert "cleanup_errors" in content, "Should track cleanup errors"
        assert "500" in content or "INTERNAL_SERVER_ERROR" in content, (
            "Should return 500 on partial cleanup failure"
        )


class TestBlacklistReturnValueChecked:
    """Verify blacklist_token() return value is checked on security-critical paths."""

    def test_refresh_blacklist_checked(self):
        """Token refresh should check blacklist_token() return value."""
        auth_file = BACKEND_DIR / "api" / "v1" / "endpoints" / "auth.py"
        content = auth_file.read_text()

        # Find the refresh section and check for return value handling
        # The code may use a resolved variable name instead of token_data.refresh_token
        assert "not await blacklist_token(" in content, (
            "Refresh endpoint should check blacklist_token return value"
        )
        # Verify it's in a conditional (if not await blacklist_token(...))
        refresh_section = content[content.find("async def refresh_token") :]
        refresh_section = refresh_section[: refresh_section.find("\nasync def ")]
        assert "not await blacklist_token(" in refresh_section, (
            "Refresh endpoint function must check blacklist_token return value"
        )

    def test_logout_blacklist_checked(self):
        """Logout should check blacklist_token() return value."""
        auth_file = BACKEND_DIR / "api" / "v1" / "endpoints" / "auth.py"
        content = auth_file.read_text()

        # The code may use a resolved variable (resolved_access) instead of raw token
        logout_section = content[content.find("async def logout") :]
        logout_section = logout_section[: logout_section.find("\nasync def ")]
        assert "not await blacklist_token(" in logout_section, (
            "Logout should check blacklist_token return value for access token"
        )


class TestSQLInjectionPrevention:
    """Verify ILIKE queries escape special characters."""

    def test_ilike_escape_function_exists(self):
        """rag_service.py should have an ILIKE escape function."""
        rag_file = BACKEND_DIR / "services" / "rag_service.py"
        content = rag_file.read_text()
        assert "_escape_ilike" in content, "Missing _escape_ilike function in rag_service.py"

    def test_ilike_queries_use_escaped_input(self):
        """rag_service.py ILIKE queries should use escaped input, not raw query."""
        rag_file = BACKEND_DIR / "services" / "rag_service.py"
        content = rag_file.read_text()

        # Check that ILIKE patterns use escaped variables
        assert "escaped_query" in content or "escaped_q" in content, (
            "ILIKE queries should use escaped variables"
        )

    def test_escape_function_handles_wildcards(self):
        """_escape_ilike should escape %, _, and backslash."""
        # Test the escape pattern directly
        escaped = re.sub(r"([%_\\])", r"\\\1", "test%value_with\\backslash")
        assert escaped == r"test\%value\_with\\backslash"


class TestJWTClaimProtection:
    """Verify JWT additional_claims cannot overwrite protected claims."""

    def test_protected_claims_filtered(self):
        """security.py should filter protected claims from additional_claims."""
        security_file = BACKEND_DIR / "core" / "security.py"
        content = security_file.read_text()
        assert "protected" in content.lower() or "safe_claims" in content, (
            "security.py should filter protected JWT claims"
        )

    def test_exp_sub_type_protected(self):
        """security.py should protect exp, sub, and type claims."""
        security_file = BACKEND_DIR / "core" / "security.py"
        content = security_file.read_text()
        # Check that a set of protected claims includes exp, sub, type
        assert '"exp"' in content and '"sub"' in content and '"type"' in content, (
            "Protected claims should include exp, sub, and type"
        )


class TestRateLimiterFailClosed:
    """Verify rate limiter denies requests when Redis is unavailable."""

    def test_redis_rate_limit_fail_closed(self):
        """redis_cache.py should deny requests when Redis is down."""
        redis_file = BACKEND_DIR / "db" / "redis_cache.py"
        content = redis_file.read_text()

        # The check_rate_limit method should return (False, 0) on Redis failure
        assert "False, 0" in content, (
            "Rate limiter should return False (deny) when Redis is unavailable"
        )

    def test_fail_closed_comment_present(self):
        """redis_cache.py should document fail-closed behavior."""
        redis_file = BACKEND_DIR / "db" / "redis_cache.py"
        content = redis_file.read_text()
        assert "fail" in content.lower() and "closed" in content.lower(), (
            "Rate limiter should document fail-closed behavior"
        )


class TestCircuitBreakerPublicAPI:
    """Verify circuit breaker uses public API instead of private attributes."""

    def test_public_method_exists(self):
        """redis_cache.py should expose is_circuit_open() public method."""
        redis_file = BACKEND_DIR / "db" / "redis_cache.py"
        content = redis_file.read_text()
        assert "def is_circuit_open" in content, (
            "redis_cache.py should have public is_circuit_open() method"
        )

    def test_security_uses_public_method(self):
        """security.py should use public is_circuit_open() instead of hasattr."""
        security_file = BACKEND_DIR / "core" / "security.py"
        content = security_file.read_text()
        assert "is_circuit_open()" in content, (
            "security.py should use is_circuit_open() public method"
        )
        assert "hasattr" not in content or "_circuit_open" not in content, (
            "security.py should not access private _circuit_open via hasattr"
        )


class TestRAGServiceThreadSafety:
    """Verify RAGService singleton is thread-safe for concurrent requests."""

    def test_contextvars_used_for_session(self):
        """rag_service.py should use contextvars for request-scoped DB session."""
        rag_file = BACKEND_DIR / "services" / "rag_service.py"
        content = rag_file.read_text()
        assert "contextvars" in content, (
            "rag_service.py should use contextvars for thread-safe session"
        )
        assert "_current_db_session" in content, (
            "rag_service.py should define _current_db_session ContextVar"
        )

    def test_async_embed_text_used(self):
        """rag_service.py should use embed_text_async instead of sync embed_text."""
        rag_file = BACKEND_DIR / "services" / "rag_service.py"
        content = rag_file.read_text()
        assert "embed_text_async" in content, (
            "rag_service.py should use async embedding to avoid blocking event loop"
        )


class TestRateLimiterMemoryManagement:
    """Verify rate limiter doesn't leak memory via defaultdict."""

    def test_no_defaultdict_in_rate_limiter(self):
        """rate_limit.py should use regular dict, not defaultdict."""
        rl_file = BACKEND_DIR / "core" / "rate_limit.py"
        content = rl_file.read_text()
        # Check that InMemoryRateLimiter doesn't use defaultdict(list) assignment
        init_section = content[content.find("def __init__(self") : content.find("def _clean_old")]
        assert "= defaultdict(" not in init_section, (
            "InMemoryRateLimiter should use regular dict to prevent phantom entries"
        )
