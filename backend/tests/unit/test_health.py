"""Build-identity tests for the health endpoints.

Why this file exists: "the site is up" is not evidence that a deploy landed.
Railway only cuts traffic over to a new container once its healthcheck passes,
so a FAILED deploy leaves the previous container serving happily behind a green
``/health``. The only thing that distinguishes the two is the commit SHA, and
until now the API reported a hardcoded ``version: "0.1.0"`` that had never
changed - forcing an operator to fingerprint the frontend JS bundle hash across
several observations to guess which build was live.

The contract these tests pin:

* the commit is answerable from ONE unauthenticated GET,
* when it is genuinely unknown the API says ``"unknown"`` - never an empty
  string, never ``None``, and never a stale-but-plausible SHA. A confident wrong
  answer is the specific failure mode this whole feature exists to prevent.
"""

import importlib.util
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.api.v1.endpoints.health as health_mod
from app.core.config import Settings, UNKNOWN_BUILD, _known

REAL_SHA = "67c6adf1e2b3c4d5a6f70819273645abcdef0123"
OTHER_SHA = "0123456789abcdef0123456789abcdef01234567"

# The four settings fields that make up a build identity.
BUILD_FIELDS = (
    "RAILWAY_GIT_COMMIT_SHA",
    "RAILWAY_GIT_BRANCH",
    "RAILWAY_DEPLOYMENT_ID",
    "COMMIT_SHA",
)


def _settings(**overrides) -> Settings:
    """A hermetic Settings instance.

    ``_env_file=None`` matters: without it pydantic-settings would read a
    developer's .env (or a CI runner that genuinely has RAILWAY_GIT_COMMIT_SHA
    exported), and the absent-value tests below would pass or fail depending on
    the machine rather than on the code.
    """
    # Every build field defaults to None here, then `overrides` replaces the
    # ones a test cares about - so a test only ever states what it is testing.
    fields = dict.fromkeys(BUILD_FIELDS, None)
    fields.update(overrides)
    return Settings(
        _env_file=None,
        SECRET_KEY="s" * 32,
        JWT_SECRET_KEY="j" * 32,
        **fields,
    )


@pytest.fixture
def build_env(monkeypatch):
    """Set the build-identity fields on the live settings singleton.

    health.py holds a reference to that singleton, so patching its fields is
    what the endpoints actually read.
    """

    def _apply(**values):
        for field in BUILD_FIELDS:
            monkeypatch.setattr(health_mod.settings, field, values.get(field))

    return _apply


@pytest.fixture
def live_client():
    """A bare app carrying only the health router.

    Isolated from the full middleware stack on purpose: these tests are about
    the endpoint's payload, and an unrelated middleware failure should not be
    able to masquerade as a build-identity regression.
    """
    application = FastAPI()
    application.include_router(health_mod.router, prefix="/api/v1/health")
    return TestClient(application, raise_server_exceptions=False)


# =============================================================================
# Resolution rules (app/core/config.py)
# =============================================================================


class TestCommitResolution:
    def test_railway_runtime_variable_is_reported(self):
        """The zero-plumbing path: Railway injects this on every deployment."""
        settings = _settings(RAILWAY_GIT_COMMIT_SHA=REAL_SHA)

        assert settings.build_commit_sha == REAL_SHA
        assert settings.build_commit_source == "railway"

    def test_build_arg_is_the_fallback_for_non_railway_images(self):
        """cd.yml's GHCR image and local `docker build` have no RAILWAY_* vars."""
        settings = _settings(COMMIT_SHA=REAL_SHA)

        assert settings.build_commit_sha == REAL_SHA
        assert settings.build_commit_source == "build-arg"

    def test_railway_wins_over_a_baked_build_arg(self):
        """The runtime variable is injected fresh; the ARG is frozen in a layer
        that a Docker cache hit could carry forward from an older commit."""
        settings = _settings(RAILWAY_GIT_COMMIT_SHA=REAL_SHA, COMMIT_SHA=OTHER_SHA)

        assert settings.build_commit_sha == REAL_SHA
        assert settings.build_commit_source == "railway"

    def test_absent_everywhere_reports_unknown(self):
        """THE case that must never lie: local dev, docker-compose, CI."""
        settings = _settings()

        assert settings.build_commit_sha == UNKNOWN_BUILD
        assert settings.build_commit_source == UNKNOWN_BUILD
        assert settings.build_branch == UNKNOWN_BUILD
        assert settings.build_deployment_id == UNKNOWN_BUILD

    @pytest.mark.parametrize("placeholder", ["", "   ", "unknown", "UNKNOWN", "None", "null"])
    def test_present_but_uninformative_values_count_as_absent(self, placeholder):
        """Each of these is something a real pipeline produces: an unexpanded
        `${GITHUB_SHA}`, a Dockerfile ARG default, a Python-formatted None.
        None of them may be echoed back as if it were a commit."""
        settings = _settings(RAILWAY_GIT_COMMIT_SHA=placeholder)

        assert settings.build_commit_sha == UNKNOWN_BUILD
        assert settings.build_commit_source == UNKNOWN_BUILD

    def test_an_uninformative_railway_value_falls_through_to_the_build_arg(self):
        """Railway present-but-empty must not shadow a real baked SHA - that
        would turn a knowable answer into "unknown"."""
        settings = _settings(RAILWAY_GIT_COMMIT_SHA="", COMMIT_SHA=REAL_SHA)

        assert settings.build_commit_sha == REAL_SHA
        assert settings.build_commit_source == "build-arg"

    def test_surrounding_whitespace_is_stripped(self):
        """A trailing newline from a shell `$(git rev-parse HEAD)` must not make
        the SHA fail a string comparison against the merge SHA."""
        assert _settings(RAILWAY_GIT_COMMIT_SHA=f"  {REAL_SHA}\n").build_commit_sha == REAL_SHA

    def test_branch_and_deployment_id_are_reported_when_railway_supplies_them(self):
        settings = _settings(RAILWAY_GIT_BRANCH="main", RAILWAY_DEPLOYMENT_ID="dep-123")

        assert settings.build_branch == "main"
        assert settings.build_deployment_id == "dep-123"

    @pytest.mark.parametrize("value", [None, "", "  ", "unknown"])
    def test_known_helper_rejects_every_uninformative_spelling(self, value):
        assert _known(value) is None

    def test_known_helper_keeps_a_real_value(self):
        assert _known(f" {REAL_SHA} ") == REAL_SHA


# =============================================================================
# /health/live - the unauthenticated answer
# =============================================================================


class TestLivenessBuildIdentity:
    def test_reports_the_commit_without_any_credentials(self, live_client, build_env):
        """One unauthenticated GET must answer "which build is live?"."""
        build_env(
            RAILWAY_GIT_COMMIT_SHA=REAL_SHA,
            RAILWAY_GIT_BRANCH="main",
            RAILWAY_DEPLOYMENT_ID="dep-123",
        )

        response = live_client.get("/api/v1/health/live")

        assert response.status_code == 200
        build = response.json()["build"]
        assert build["commit"] == REAL_SHA
        assert build["commit_source"] == "railway"
        assert build["branch"] == "main"
        assert build["deployment_id"] == "dep-123"

    def test_absent_env_vars_report_unknown_and_do_not_crash(self, live_client, build_env):
        """The regression that matters most: no env var, no exception, no
        fabricated value, and nothing that could be mistaken for a real SHA."""
        build_env()

        response = live_client.get("/api/v1/health/live")

        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "alive"
        assert payload["build"] == {
            "commit": UNKNOWN_BUILD,
            "commit_source": UNKNOWN_BUILD,
            "branch": UNKNOWN_BUILD,
            "deployment_id": UNKNOWN_BUILD,
        }

    def test_no_field_is_ever_null_or_empty(self, live_client, build_env):
        """`null` in JSON would push the "is it deployed?" decision onto every
        consumer. "unknown" is a value; None is a hole."""
        build_env()

        build = live_client.get("/api/v1/health/live").json()["build"]

        assert all(isinstance(v, str) and v for v in build.values())

    def test_uptime_is_reported_and_measured_from_process_start(
        self, live_client, build_env, monkeypatch
    ):
        """Uptime answers "did a new container actually start?" even when the
        commit is unknown - a process up for an hour did not just take your
        merge."""
        build_env()
        monkeypatch.setattr(health_mod, "_startup_time", time.time() - 3600)

        payload = live_client.get("/api/v1/health/live").json()

        assert payload["uptime_seconds"] >= 3600


class TestRootHealthLiveProxy:
    """/health/live at the ROOT is the URL an operator actually curls.

    app/main.py exposes it as a thin proxy that awaits liveness_check(), so the
    build identity reaches it for free - but only as long as that delegation
    survives. If someone ever inlines a literal dict there (as the root /health
    endpoint still does), this test is what notices.
    """

    def test_root_health_live_carries_the_build_identity(self, build_env):
        from app.main import create_application

        build_env(RAILWAY_GIT_COMMIT_SHA=REAL_SHA)
        client = TestClient(create_application(), raise_server_exceptions=False)

        response = client.get("/health/live")

        assert response.status_code == 200
        assert response.json()["build"]["commit"] == REAL_SHA

    def test_root_health_live_is_never_cached(self, build_env):
        """A shared cache serving a stale body would reintroduce the exact bug:
        a confident answer describing a container that is no longer running."""
        from app.main import create_application

        build_env(RAILWAY_GIT_COMMIT_SHA=REAL_SHA)
        client = TestClient(create_application(), raise_server_exceptions=False)

        response = client.get("/health/live")

        assert "no-store" in response.headers.get("Cache-Control", "")


class TestStartupTimeCapture:
    def test_startup_time_is_captured_at_import_not_on_first_call(self):
        """Regression: it used to initialize lazily, so uptime was measured from
        "the first time anyone asked". A container up for days then reported ~0s
        to whoever checked first - reading as a fresh deploy at the exact moment
        you needed proof of one.

        Loaded under a throwaway module name so the assertion is about a module
        that has never been called, independent of test ordering.
        """
        spec = importlib.util.spec_from_file_location("_health_fresh", health_mod.__file__)
        fresh = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fresh)

        assert isinstance(fresh._startup_time, float)
        assert fresh.get_startup_time() == fresh._startup_time


# =============================================================================
# /health/detailed - the hardcoded version is gone
# =============================================================================


class TestDetailedBuildIdentity:
    @pytest.mark.asyncio
    async def test_detailed_reports_the_real_commit_instead_of_a_frozen_version(
        self, build_env, monkeypatch
    ):
        """`version: "0.1.0"` shipped unchanged through every release, so it
        could not distinguish two builds. It is replaced, not merely updated."""
        build_env(RAILWAY_GIT_COMMIT_SHA=REAL_SHA)
        monkeypatch.setattr(health_mod, "update_system_metrics", lambda: None)
        for check in (
            "check_postgres_health",
            "check_neo4j_health",
            "check_qdrant_health",
            "check_redis_health",
            "_check_embedding_health_bounded",
        ):
            monkeypatch.setattr(
                health_mod,
                check,
                _healthy_stub(check),
            )

        response = await health_mod.detailed_health_check(current_user=None)

        assert response.build.commit == REAL_SHA
        assert response.build.commit_source == "railway"

    def test_the_response_model_no_longer_carries_a_hardcoded_version_field(self):
        assert "version" not in health_mod.DetailedHealthResponse.model_fields
        assert "build" in health_mod.DetailedHealthResponse.model_fields


def _healthy_stub(name: str):
    """An async no-op health check that always reports healthy."""

    async def _stub():
        return health_mod.ServiceHealth(name=name, status="healthy")

    return _stub
