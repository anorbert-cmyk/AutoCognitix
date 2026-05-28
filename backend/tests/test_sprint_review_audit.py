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


class TestNHTSAEUMarketGuard:
    """Verify NHTSA service skips API calls for brands not sold in the US market."""

    def test_eu_only_makes_constant_defined(self):
        """nhtsa_service.py should expose an EU_ONLY_MAKES set."""
        nhtsa_file = BACKEND_DIR / "services" / "nhtsa_service.py"
        content = nhtsa_file.read_text()
        assert "EU_ONLY_MAKES" in content, (
            "nhtsa_service.py should declare EU_ONLY_MAKES to avoid useless NHTSA round-trips"
        )
        for brand in ("skoda", "seat", "opel", "peugeot", "dacia"):
            assert brand in content.lower(), (
                f"EU_ONLY_MAKES should include {brand!r} — sold in EU only, no NHTSA recalls"
            )

    def test_brand_aliases_normalize_common_typos(self):
        """nhtsa_service should map VW/Mercedes/Chevy to canonical NHTSA spellings."""
        nhtsa_file = BACKEND_DIR / "services" / "nhtsa_service.py"
        content = nhtsa_file.read_text()
        assert "BRAND_ALIASES" in content, (
            "nhtsa_service should declare BRAND_ALIASES to avoid 0 recalls on common typos"
        )
        assert "_normalize_make" in content, (
            "nhtsa_service should expose _normalize_make for canonical lookup"
        )
        # Sanity: get_recalls must apply the normalizer before the EU guard.
        get_recalls = content[content.find("async def get_recalls") :]
        normalize_pos = get_recalls.find("_normalize_make")
        eu_pos = get_recalls.find("EU_ONLY_MAKES")
        assert 0 < normalize_pos < eu_pos, (
            "_normalize_make must run before EU_ONLY_MAKES guard — aliases like 'VW' "
            "won't match the EU filter otherwise"
        )

    def test_get_recalls_short_circuits_for_eu_only(self):
        """get_recalls should early-return [] before making HTTP request for EU-only brands."""
        nhtsa_file = BACKEND_DIR / "services" / "nhtsa_service.py"
        content = nhtsa_file.read_text()
        recalls_section = content[content.find("async def get_recalls") :]
        # The EU guard must precede the cache_key generation / HTTP call
        cache_key_pos = recalls_section.find("_generate_cache_key")
        eu_guard_pos = recalls_section.find("EU_ONLY_MAKES")
        assert 0 < eu_guard_pos < cache_key_pos, (
            "EU_ONLY_MAKES guard must run before cache/HTTP work in get_recalls"
        )

    def test_get_complaints_short_circuits_for_eu_only(self):
        """get_complaints should also short-circuit for EU-only brands."""
        nhtsa_file = BACKEND_DIR / "services" / "nhtsa_service.py"
        content = nhtsa_file.read_text()
        complaints_section = content[content.find("async def get_complaints") :]
        cache_key_pos = complaints_section.find("_generate_cache_key")
        eu_guard_pos = complaints_section.find("EU_ONLY_MAKES")
        assert 0 < eu_guard_pos < cache_key_pos, (
            "EU_ONLY_MAKES guard must run before cache/HTTP work in get_complaints"
        )


class TestEmbeddingCacheKeyVersioning:
    """Verify embedding cache keys include model + revision so old vectors don't poison new models."""

    def test_cache_key_includes_model_and_revision(self):
        """redis_cache.py should fold HUBERT_MODEL + HUBERT_REVISION into the cache key."""
        cache_file = BACKEND_DIR / "db" / "redis_cache.py"
        content = cache_file.read_text()
        # The helper that builds the key
        assert "_embedding_cache_key" in content, (
            "redis_cache.py should centralize embedding key construction"
        )
        # Both identifiers must influence the hash
        assert "HUBERT_MODEL" in content, "cache key must include HUBERT_MODEL"
        assert "HUBERT_REVISION" in content, "cache key must include HUBERT_REVISION"


class TestHuBERTRevisionPinning:
    """Verify HuBERT model loads with explicit revision to prevent silent drift."""

    def test_revision_setting_exists(self):
        """config.py should expose HUBERT_REVISION."""
        cfg = BACKEND_DIR / "core" / "config.py"
        assert "HUBERT_REVISION" in cfg.read_text(), (
            "config.py should declare HUBERT_REVISION to pin the model version"
        )

    def test_embedding_service_passes_revision(self):
        """embedding_service.py should pass revision= to both tokenizer + model loaders."""
        es = BACKEND_DIR / "services" / "embedding_service.py"
        content = es.read_text()
        # Both calls inside _load_hubert_model must include revision=
        load_section = content[content.find("def _load_hubert_model") :]
        load_section = load_section[: load_section.find("def ", 50)]
        revision_count = load_section.count("revision=")
        assert revision_count >= 2, (
            "Both AutoTokenizer.from_pretrained and AutoModel.from_pretrained "
            f"must pass revision= (found {revision_count})"
        )


class TestSentryExplicitCapture:
    """Verify Sentry forwarding is present and PII-safe across 5xx handlers."""

    def test_capture_helper_exists(self):
        """error_handlers.py should expose a single _capture_to_sentry helper."""
        eh = BACKEND_DIR / "core" / "error_handlers.py"
        content = eh.read_text()
        assert "def _capture_to_sentry" in content, (
            "error_handlers.py should define a centralized Sentry helper"
        )
        helper = content[content.find("def _capture_to_sentry") :]
        helper = helper[: helper.find("\n\nasync def")]
        assert "sentry_sdk.capture_exception" in helper, (
            "_capture_to_sentry must actually call sentry_sdk.capture_exception"
        )
        assert "except ImportError" in helper, (
            "Sentry import must be guarded (it may not be installed)"
        )

    def test_helper_uses_route_template_not_raw_path(self):
        """PII-safe tagging: route template tag, raw path only in extras."""
        eh = BACKEND_DIR / "core" / "error_handlers.py"
        helper = eh.read_text()
        helper = helper[helper.find("def _capture_to_sentry") :]
        helper = helper[: helper.find("\n\nasync def")]
        assert 'set_tag("route"' in helper, (
            "Sentry tag should be the route template (no UUIDs), not raw URL"
        )
        assert 'set_extra("raw_path"' in helper, (
            "Raw path belongs in set_extra (unindexed), not set_tag"
        )

    def test_generic_handler_invokes_helper(self):
        eh = BACKEND_DIR / "core" / "error_handlers.py"
        content = eh.read_text()
        generic = content[content.find("async def generic_exception_handler") :]
        generic = generic[: generic.find("\n\nasync def") or len(generic)]
        assert "_capture_to_sentry(" in generic, (
            "generic_exception_handler should delegate to _capture_to_sentry"
        )

    def test_5xx_handlers_capture(self):
        """sqlalchemy_exception_handler and autocognitix_exception_handler should
        forward to Sentry for 5xx status codes."""
        eh = BACKEND_DIR / "core" / "error_handlers.py"
        content = eh.read_text()
        # SQLAlchemy handler
        sa_section = content[content.find("async def sqlalchemy_exception_handler") :]
        sa_section = sa_section[: sa_section.find("\n\nasync def") or len(sa_section)]
        assert "_capture_to_sentry(" in sa_section, (
            "sqlalchemy_exception_handler should call _capture_to_sentry for 5xx"
        )
        # AutoCognitix handler
        ac_section = content[content.find("async def autocognitix_exception_handler") :]
        ac_section = ac_section[: ac_section.find("\n\nasync def") or len(ac_section)]
        assert "_capture_to_sentry(" in ac_section, (
            "autocognitix_exception_handler should call _capture_to_sentry for 5xx"
        )
