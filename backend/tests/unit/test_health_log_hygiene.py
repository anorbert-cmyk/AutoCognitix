"""Log-hygiene regression tests for the embedding health probe.

The project rule (CLAUDE.md, tasks/lessons.md) is unconditional: every value
that enters a log record - including numbers - goes through ``sanitize_log``,
and exceptions through ``sanitize_exception``. CWE-117: a raw ``\\n`` in a
logged value forges a whole log line, and the log is the GDPR/security audit
trail.

``check_embedding_health`` and ``_check_embedding_health_bounded`` shipped
without it in three places. Everything they log is attacker-influenceable in
the way that matters: the probe's ``error`` field is built inside
``embedding_self_test`` as ``f"{type(e).__name__}: {e}"`` from an arbitrary
backend exception - a model path, a tokenizer file, an HTTP body.

Each test asserts on the RECORD ATTRIBUTE rather than the rendered line,
because ``extra={}`` values bypass %-formatting entirely and reach the
structured JSON handler verbatim.
"""

import asyncio
import logging

import pytest

import app.api.v1.endpoints.health as health_mod

SELF_TEST = "app.services.embedding_service.embedding_self_test"

# One forged log line, the classic CWE-117 payload.
INJECTION = "boom\nCRITICAL: root login from 10.0.0.1"


def _record(caplog, event: str) -> logging.LogRecord:
    """The single record carrying `event`, or fail with a useful message."""
    matches = [r for r in caplog.records if getattr(r, "event", None) == event]
    assert len(matches) == 1, f"expected exactly one {event!r} record, got {len(matches)}"
    return matches[0]


class TestEmbeddingHealthLogHygiene:
    @pytest.mark.asyncio
    async def test_a_probe_error_string_is_sanitized(self, caplog, monkeypatch):
        """`error` comes straight from embedding_self_test and is unbounded."""
        monkeypatch.setattr(
            SELF_TEST,
            lambda: {"status": "unavailable", "backend": "onnx", "error": INJECTION},
        )

        with caplog.at_level(logging.ERROR):
            await health_mod.check_embedding_health()

        record = _record(caplog, "embedding_backend_unusable")
        assert "\n" not in record.error
        assert "\\n" in record.error

    @pytest.mark.asyncio
    async def test_probe_status_and_backend_are_sanitized_too(self, caplog, monkeypatch):
        """They look like fixed enums; they are whatever the probe returned."""
        monkeypatch.setattr(
            SELF_TEST,
            lambda: {"status": "degraded", "backend": INJECTION, "error": None},
        )

        with caplog.at_level(logging.ERROR):
            await health_mod.check_embedding_health()

        record = _record(caplog, "embedding_backend_unusable")
        assert "\n" not in record.backend
        assert isinstance(record.probe_status, str)

    @pytest.mark.asyncio
    async def test_the_response_body_still_carries_the_real_error(self, caplog, monkeypatch):
        """Sanitizing the LOG must not degrade what the operator sees in JSON."""
        monkeypatch.setattr(
            SELF_TEST,
            lambda: {"status": "unavailable", "backend": None, "error": "no backend"},
        )

        with caplog.at_level(logging.ERROR):
            result = await health_mod.check_embedding_health()

        assert result.status == "degraded"
        assert result.error == "no backend"

    @pytest.mark.asyncio
    async def test_an_exploding_probe_logs_a_sanitized_exception(self, caplog, monkeypatch):
        """The `except` arm interpolated the raw exception into the message."""

        def _explode():
            raise RuntimeError(INJECTION)

        monkeypatch.setattr(SELF_TEST, _explode)

        with caplog.at_level(logging.ERROR):
            result = await health_mod.check_embedding_health()

        assert result.status == "unhealthy"
        failures = [r for r in caplog.records if "Embedding health check failed" in r.getMessage()]
        assert len(failures) == 1
        assert "\n" not in failures[0].getMessage()
        assert "\\n" in failures[0].getMessage()

    @pytest.mark.asyncio
    async def test_the_timeout_record_sanitizes_its_number(self, caplog, monkeypatch):
        """Numbers are not exempt - the rule is deliberately unconditional, so a
        field never has to be re-audited when its source becomes configurable."""

        async def _never_finishes():
            await asyncio.sleep(3600)

        monkeypatch.setattr(health_mod, "EMBEDDING_HEALTH_TIMEOUT_SECONDS", 0.01)
        monkeypatch.setattr(health_mod, "check_embedding_health", _never_finishes)

        with caplog.at_level(logging.ERROR):
            result = await health_mod._check_embedding_health_bounded()

        assert result.status == "degraded"
        record = _record(caplog, "embedding_health_timeout")
        assert isinstance(record.timeout_seconds, str), (
            "sanitize_log returns a str; a raw float here means the value never passed through it"
        )
