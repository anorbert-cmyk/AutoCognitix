"""Sprint 9 Critical Fix Tests.

Tests for critical security, database, and business logic fixes.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError


# Path to source files for file-based inspection
BACKEND_APP_DIR = Path(__file__).resolve().parent.parent / "app"


def _read_source_file(relative_path: str) -> str:
    """Read a source file and return its contents as a string.

    Used instead of inspect.getsource() when modules cannot be imported
    due to missing dependencies (sqlalchemy, qdrant-client, etc.).
    """
    filepath = BACKEND_APP_DIR / relative_path
    return filepath.read_text(encoding="utf-8")


class TestTimezoneAwareness:
    """Verify all datetime operations use timezone-aware timestamps."""

    def test_no_utcnow_in_repositories(self):
        """Ensure repositories.py doesn't use datetime.utcnow()."""
        source = _read_source_file("db/postgres/repositories.py")
        assert "utcnow()" not in source, (
            "Found datetime.utcnow() in repositories.py - use datetime.now(timezone.utc) instead"
        )

    def test_no_utcnow_in_diagnosis_service(self):
        """Ensure diagnosis_service.py doesn't use datetime.utcnow()."""
        source = _read_source_file("services/diagnosis_service.py")
        assert "utcnow()" not in source, (
            "Found datetime.utcnow() in diagnosis_service.py - use datetime.now(timezone.utc) instead"
        )


class TestConfidenceScoreValidation:
    """Verify confidence scores are always clamped to [0.0, 1.0]."""

    def test_probable_cause_rejects_high_confidence(self):
        """ProbableCause should reject confidence > 1.0 via Pydantic constraint."""
        from app.api.v1.schemas.diagnosis import ProbableCause

        with pytest.raises(ValidationError, match="less than or equal"):
            ProbableCause(
                title="Test cause",
                description="Test leírás",
                confidence=1.5,
            )

    def test_probable_cause_rejects_negative_confidence(self):
        """ProbableCause should reject confidence < 0.0 via Pydantic constraint."""
        from app.api.v1.schemas.diagnosis import ProbableCause

        with pytest.raises(ValidationError, match="greater than or equal"):
            ProbableCause(
                title="Test cause",
                description="Test leírás",
                confidence=-0.5,
            )

    def test_probable_cause_valid_confidence_unchanged(self):
        """ProbableCause should leave valid confidence values unchanged."""
        from app.api.v1.schemas.diagnosis import ProbableCause

        cause = ProbableCause(
            title="Test cause",
            description="Test leírás",
            confidence=0.85,
        )
        assert cause.confidence == 0.85

    def test_probable_cause_boundary_zero(self):
        """ProbableCause should accept confidence = 0.0."""
        from app.api.v1.schemas.diagnosis import ProbableCause

        cause = ProbableCause(
            title="Test cause",
            description="Test leírás",
            confidence=0.0,
        )
        assert cause.confidence == 0.0

    def test_probable_cause_boundary_one(self):
        """ProbableCause should accept confidence = 1.0."""
        from app.api.v1.schemas.diagnosis import ProbableCause

        cause = ProbableCause(
            title="Test cause",
            description="Test leírás",
            confidence=1.0,
        )
        assert cause.confidence == 1.0

    def test_source_rejects_high_relevance_score(self):
        """Source should reject relevance_score > 1.0."""
        from app.api.v1.schemas.diagnosis import Source

        with pytest.raises(ValidationError, match="less than or equal"):
            Source(
                type="database",
                title="Test source",
                relevance_score=2.0,
            )

    def test_source_rejects_negative_relevance_score(self):
        """Source should reject relevance_score < 0.0."""
        from app.api.v1.schemas.diagnosis import Source

        with pytest.raises(ValidationError, match="greater than or equal"):
            Source(
                type="database",
                title="Test source",
                relevance_score=-0.3,
            )

    def test_source_valid_relevance_score(self):
        """Source should accept valid relevance_score in [0.0, 1.0]."""
        from app.api.v1.schemas.diagnosis import Source

        source = Source(
            type="database",
            title="Test source",
            relevance_score=0.75,
        )
        assert source.relevance_score == 0.75


class TestCostEstimateValidation:
    """Verify cost estimates have min <= max."""

    def test_valid_cost_range(self):
        """parts_min <= parts_max should pass validation."""
        from app.api.v1.schemas.diagnosis import TotalCostEstimate

        estimate = TotalCostEstimate(
            parts_min=10000,
            parts_max=20000,
            labor_min=5000,
            labor_max=10000,
            total_min=15000,
            total_max=30000,
            currency="HUF",
        )
        assert estimate.parts_min <= estimate.parts_max
        assert estimate.labor_min <= estimate.labor_max

    def test_equal_min_max_is_valid(self):
        """parts_min == parts_max should pass validation."""
        from app.api.v1.schemas.diagnosis import TotalCostEstimate

        estimate = TotalCostEstimate(
            parts_min=15000,
            parts_max=15000,
            labor_min=5000,
            labor_max=5000,
            total_min=20000,
            total_max=20000,
        )
        assert estimate.parts_min == estimate.parts_max

    def test_invalid_cost_range_parts_swapped(self):
        """parts_min > parts_max should raise ValidationError."""
        from app.api.v1.schemas.diagnosis import TotalCostEstimate

        with pytest.raises(ValidationError, match="parts_min"):
            TotalCostEstimate(
                parts_min=30000,
                parts_max=10000,
                labor_min=5000,
                labor_max=10000,
                total_min=35000,
                total_max=20000,
            )

    def test_invalid_cost_range_labor_swapped(self):
        """labor_min > labor_max should raise ValidationError."""
        from app.api.v1.schemas.diagnosis import TotalCostEstimate

        with pytest.raises(ValidationError, match="labor_min"):
            TotalCostEstimate(
                parts_min=10000,
                parts_max=20000,
                labor_min=15000,
                labor_max=5000,
                total_min=25000,
                total_max=25000,
            )


class TestRateLimitConfig:
    """Verify rate limiting is strict enough.

    Uses file-based source inspection since app.core.rate_limit requires
    pydantic-settings and fastapi which may not be installed in test env.
    """

    def test_auth_rate_limit_strict(self):
        """Auth rate limit should be max 5 requests per minute."""
        source = _read_source_file("core/rate_limit.py")
        # Parse AUTH_CONFIG definition to check requests_per_minute
        assert "AUTH_CONFIG = RateLimitConfig(" in source, "AUTH_CONFIG not found in rate_limit.py"
        # Verify the requests_per_minute value in AUTH_CONFIG block
        import re

        match = re.search(
            r"AUTH_CONFIG\s*=\s*RateLimitConfig\([^)]*requests_per_minute\s*=\s*(\d+)",
            source,
            re.DOTALL,
        )
        assert match, "Could not find requests_per_minute in AUTH_CONFIG"
        rpm = int(match.group(1))
        assert rpm <= 5, f"Auth rate limit too permissive: {rpm}/min"

    def test_auth_lockout_threshold_strict(self):
        """Auth lockout should trigger at max 5 failed attempts."""
        source = _read_source_file("core/rate_limit.py")
        import re

        match = re.search(
            r"AUTH_CONFIG\s*=\s*RateLimitConfig\([^)]*lockout_threshold\s*=\s*(\d+)",
            source,
            re.DOTALL,
        )
        assert match, "Could not find lockout_threshold in AUTH_CONFIG"
        threshold = int(match.group(1))
        assert threshold <= 5, f"Lockout threshold too permissive: {threshold}"

    def test_auth_lockout_duration_minimum(self):
        """Auth lockout duration should be at least 5 minutes (300 seconds)."""
        source = _read_source_file("core/rate_limit.py")
        import re

        match = re.search(
            r"AUTH_CONFIG\s*=\s*RateLimitConfig\([^)]*lockout_duration_seconds\s*=\s*(\d+)",
            source,
            re.DOTALL,
        )
        assert match, "Could not find lockout_duration_seconds in AUTH_CONFIG"
        duration = int(match.group(1))
        assert duration >= 300, f"Lockout duration too short: {duration}s"


class TestEmbeddingVersionTracking:
    """Verify embedding model version is tracked.

    Uses file-based source inspection since app.db.qdrant_client requires
    qdrant-client which may not be installed in test env.
    """

    def test_qdrant_service_has_version_constant(self):
        """QdrantService should define EMBEDDING_MODEL_VERSION."""
        source = _read_source_file("db/qdrant_client.py")
        assert "EMBEDDING_MODEL_VERSION" in source, (
            "QdrantService must define EMBEDDING_MODEL_VERSION class attribute"
        )

    def test_qdrant_service_version_is_nonempty(self):
        """EMBEDDING_MODEL_VERSION should be assigned a non-empty string."""
        source = _read_source_file("db/qdrant_client.py")
        import re

        match = re.search(
            r'EMBEDDING_MODEL_VERSION\s*=\s*["\'](.+?)["\']',
            source,
        )
        assert match, "EMBEDDING_MODEL_VERSION must be assigned a non-empty string value"
        version = match.group(1)
        assert len(version) > 0, "EMBEDDING_MODEL_VERSION must not be empty"

    def test_qdrant_service_has_expected_dimension(self):
        """QdrantService should define EXPECTED_DIMENSION as 768."""
        source = _read_source_file("db/qdrant_client.py")
        import re

        match = re.search(
            r"EXPECTED_DIMENSION\s*=\s*(\d+)",
            source,
        )
        assert match, "QdrantService must define EXPECTED_DIMENSION class attribute"
        dimension = int(match.group(1))
        assert dimension == 768, f"EXPECTED_DIMENSION should be 768, got {dimension}"
