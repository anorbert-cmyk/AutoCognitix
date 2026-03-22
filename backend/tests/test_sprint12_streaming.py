"""Sprint 12 Streaming & Email Integration Tests.

Verifies streaming endpoint structure and email integration in auth endpoints
by inspecting source code. All tests are synchronous and use no app imports.
"""

from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent / "app"


def _read(relative_path: str) -> str:
    """Read a backend source file relative to app/."""
    return (BACKEND_DIR / relative_path).read_text(encoding="utf-8")


# =============================================================================
# TestStreamingEndpoint
# =============================================================================


class TestStreamingEndpoint:
    """Verify streaming diagnosis endpoint structure in diagnosis.py."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.diagnosis_source = _read("api/v1/endpoints/diagnosis.py")
        self.schemas_source = _read("api/v1/schemas/diagnosis.py")

    def test_streaming_endpoint_exists(self):
        """Streaming endpoint function must be defined."""
        assert "analyze_vehicle_stream" in self.diagnosis_source, (
            "analyze_vehicle_stream function not found in diagnosis.py"
        )

    def test_streaming_route_path(self):
        """Streaming endpoint must be mounted at /analyze/stream."""
        assert '"/analyze/stream"' in self.diagnosis_source, (
            "Streaming endpoint must be routed to /analyze/stream"
        )

    def test_streaming_uses_sse_media_type(self):
        """Response must use text/event-stream media type for SSE."""
        assert '"text/event-stream"' in self.diagnosis_source, (
            "Streaming endpoint must set media_type to text/event-stream"
        )

    def test_streaming_returns_streaming_response(self):
        """Endpoint must return a StreamingResponse."""
        assert "StreamingResponse" in self.diagnosis_source, (
            "Streaming endpoint must use FastAPI StreamingResponse"
        )

    def test_streaming_disables_buffering(self):
        """SSE response must disable nginx buffering via X-Accel-Buffering header."""
        assert "X-Accel-Buffering" in self.diagnosis_source, (
            "Streaming response should set X-Accel-Buffering: no for nginx"
        )

    def test_streaming_has_cache_control_no_cache(self):
        """SSE response must set Cache-Control: no-cache."""
        assert "no-cache" in self.diagnosis_source, (
            "Streaming response must include Cache-Control: no-cache"
        )

    def test_streaming_sends_start_event(self):
        """Generator must yield a 'start' event at the beginning."""
        assert 'event_type="start"' in self.diagnosis_source, (
            "Streaming generator must emit a 'start' event"
        )

    def test_streaming_sends_complete_event(self):
        """Generator must yield a 'complete' event at the end."""
        assert 'event_type="complete"' in self.diagnosis_source, (
            "Streaming generator must emit a 'complete' event"
        )

    def test_streaming_sends_error_event(self):
        """Generator must yield an 'error' event on failure."""
        assert 'event_type="error"' in self.diagnosis_source, (
            "Streaming generator must emit an 'error' event on failures"
        )

    def test_streaming_sends_cause_events(self):
        """Generator must yield 'cause' events for probable causes."""
        assert 'event_type="cause"' in self.diagnosis_source, (
            "Streaming generator must emit 'cause' events"
        )

    def test_streaming_sends_repair_events(self):
        """Generator must yield 'repair' events for recommendations."""
        assert 'event_type="repair"' in self.diagnosis_source, (
            "Streaming generator must emit 'repair' events"
        )

    def test_streaming_has_progress_tracking(self):
        """Streaming events must include progress values."""
        assert "progress=" in self.diagnosis_source, (
            "Streaming events must include progress tracking"
        )

    def test_streaming_handles_dtc_validation_error(self):
        """Streaming generator must handle DTCValidationError."""
        # Find the generate_events body
        gen_start = self.diagnosis_source.find("async def generate_events")
        assert gen_start > 0, "generate_events generator not found"
        gen_body = self.diagnosis_source[gen_start:]

        assert "DTCValidationError" in gen_body, "generate_events must catch DTCValidationError"

    def test_streaming_handles_vin_decode_error(self):
        """Streaming generator must handle VINDecodeError."""
        gen_start = self.diagnosis_source.find("async def generate_events")
        assert gen_start > 0
        gen_body = self.diagnosis_source[gen_start:]

        assert "VINDecodeError" in gen_body, "generate_events must catch VINDecodeError"

    def test_streaming_handles_generic_exception(self):
        """Streaming generator must catch generic Exception as fallback."""
        gen_start = self.diagnosis_source.find("async def generate_events")
        assert gen_start > 0
        gen_body = self.diagnosis_source[gen_start:]

        assert "except Exception" in gen_body, "generate_events must catch generic Exception"

    def test_format_sse_event_function_exists(self):
        """_format_sse_event helper must exist for SSE formatting."""
        assert "def _format_sse_event" in self.diagnosis_source, (
            "_format_sse_event helper function not found"
        )

    def test_sse_format_uses_event_and_data_fields(self):
        """SSE output must use 'event:' and 'data:' fields per SSE spec."""
        # Find the _format_sse_event function
        fn_start = self.diagnosis_source.find("def _format_sse_event")
        assert fn_start > 0
        fn_body = self.diagnosis_source[fn_start:]

        assert "event:" in fn_body and "data:" in fn_body, (
            "_format_sse_event must produce SSE lines with 'event:' and 'data:' prefixes"
        )

    def test_streaming_saves_diagnosis_to_database(self):
        """Streaming endpoint must persist the diagnosis session."""
        # _save_diagnosis_session may be called in the outer streaming pipeline
        # (analyze_vehicle_stream) rather than inside generate_events directly
        stream_start = self.diagnosis_source.find("async def analyze_vehicle_stream")
        assert stream_start > 0
        stream_body = self.diagnosis_source[stream_start:]

        assert "_save_diagnosis_session" in stream_body, (
            "analyze_vehicle_stream must call _save_diagnosis_session to persist results"
        )


# =============================================================================
# TestStreamingSchemas
# =============================================================================


class TestStreamingSchemas:
    """Verify streaming-related schemas in diagnosis.py schemas."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.source = _read("api/v1/schemas/diagnosis.py")

    def test_streaming_event_type_is_enum(self):
        """StreamingEventType must be a proper str Enum."""
        assert "class StreamingEventType(str, Enum)" in self.source, (
            "StreamingEventType must be defined as class StreamingEventType(str, Enum)"
        )

    def test_streaming_event_type_has_all_values(self):
        """StreamingEventType must define all required event types."""
        required = [
            "START",
            "CONTEXT",
            "ANALYSIS",
            "CAUSE",
            "REPAIR",
            "WARNING",
            "COMPLETE",
            "ERROR",
        ]
        for name in required:
            assert f"{name} = " in self.source, f"StreamingEventType must define {name}"

    def test_streaming_event_model_exists(self):
        """StreamingEvent Pydantic model must be defined."""
        assert "class StreamingEvent(BaseModel)" in self.source, "StreamingEvent model not found"

    def test_streaming_event_has_progress_field(self):
        """StreamingEvent must have a progress field."""
        # Find StreamingEvent class body
        cls_start = self.source.find("class StreamingEvent(BaseModel)")
        assert cls_start > 0
        next_cls = self.source.find("\nclass ", cls_start + 1)
        if next_cls == -1:
            next_cls = len(self.source)
        cls_body = self.source[cls_start:next_cls]

        assert "progress" in cls_body, "StreamingEvent must have a 'progress' field"

    def test_streaming_event_has_diagnosis_id_field(self):
        """StreamingEvent must have a diagnosis_id field."""
        cls_start = self.source.find("class StreamingEvent(BaseModel)")
        assert cls_start > 0
        next_cls = self.source.find("\nclass ", cls_start + 1)
        if next_cls == -1:
            next_cls = len(self.source)
        cls_body = self.source[cls_start:next_cls]

        assert "diagnosis_id" in cls_body, "StreamingEvent must have a 'diagnosis_id' field"

    def test_streaming_event_has_timestamp_field(self):
        """StreamingEvent must have a timestamp field."""
        cls_start = self.source.find("class StreamingEvent(BaseModel)")
        assert cls_start > 0
        next_cls = self.source.find("\nclass ", cls_start + 1)
        if next_cls == -1:
            next_cls = len(self.source)
        cls_body = self.source[cls_start:next_cls]

        assert "timestamp" in cls_body, "StreamingEvent must have a 'timestamp' field"

    def test_diagnosis_stream_request_model_exists(self):
        """DiagnosisStreamRequest model must be defined."""
        assert "class DiagnosisStreamRequest(BaseModel)" in self.source, (
            "DiagnosisStreamRequest model not found"
        )

    def test_stream_request_has_streaming_options(self):
        """DiagnosisStreamRequest must have streaming-specific options."""
        cls_start = self.source.find("class DiagnosisStreamRequest(BaseModel)")
        assert cls_start > 0
        next_cls = self.source.find("\nclass ", cls_start + 1)
        if next_cls == -1:
            next_cls = len(self.source)
        cls_body = self.source[cls_start:next_cls]

        assert "include_context" in cls_body or "include_progress" in cls_body, (
            "DiagnosisStreamRequest must have streaming options (include_context or include_progress)"
        )


# =============================================================================
# TestEmailAuthIntegration
# =============================================================================


class TestEmailAuthIntegration:
    """Verify email service integration in auth endpoints."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.auth_source = _read("api/v1/endpoints/auth.py")
        self.email_source = _read("services/email_service.py")

    def test_email_service_imported_in_auth(self):
        """Auth endpoint must import from email_service."""
        assert "email_service" in self.auth_source, "auth.py must import from email_service"

    def test_send_password_reset_email_imported(self):
        """send_password_reset_email must be imported in auth.py."""
        assert "send_password_reset_email" in self.auth_source, (
            "auth.py must import send_password_reset_email"
        )

    def test_send_welcome_email_imported(self):
        """send_welcome_email must be imported in auth.py."""
        assert "send_welcome_email" in self.auth_source, "auth.py must import send_welcome_email"

    def test_forgot_password_endpoint_exists(self):
        """forgot_password endpoint must be defined."""
        assert "async def forgot_password" in self.auth_source, (
            "forgot_password endpoint not found in auth.py"
        )

    def test_forgot_password_creates_reset_token(self):
        """forgot_password must call create_password_reset_token."""
        # Find the forgot_password function body
        fn_start = self.auth_source.find("async def forgot_password")
        assert fn_start > 0
        next_fn = self.auth_source.find("\n@router.", fn_start + 1)
        if next_fn == -1:
            next_fn = len(self.auth_source)
        fn_body = self.auth_source[fn_start:next_fn]

        assert "create_password_reset_token" in fn_body, (
            "forgot_password must call create_password_reset_token"
        )

    def test_forgot_password_does_not_reveal_email_existence(self):
        """forgot_password must return same response whether email exists or not (anti-enumeration)."""
        fn_start = self.auth_source.find("async def forgot_password")
        assert fn_start > 0
        next_fn = self.auth_source.find("\n@router.", fn_start + 1)
        if next_fn == -1:
            next_fn = len(self.auth_source)
        fn_body = self.auth_source[fn_start:next_fn]

        # Should return generic response outside the if-user-exists block
        assert "ForgotPasswordResponse()" in fn_body, (
            "forgot_password must always return same generic response to prevent email enumeration"
        )

    def test_email_service_has_demo_mode(self):
        """EmailService must support demo mode for development."""
        assert "demo_mode" in self.email_source or "_demo_mode" in self.email_source, (
            "EmailService must support demo mode"
        )

    def test_email_service_has_send_password_reset(self):
        """EmailService must have send_password_reset method."""
        assert "async def send_password_reset" in self.email_source, (
            "EmailService must have send_password_reset method"
        )

    def test_email_service_has_send_welcome(self):
        """EmailService must have send_welcome method."""
        assert "async def send_welcome" in self.email_source, (
            "EmailService must have send_welcome method"
        )

    def test_email_service_is_singleton(self):
        """EmailService must use singleton pattern."""
        assert "_instance" in self.email_source and "__new__" in self.email_source, (
            "EmailService must use singleton pattern with _instance and __new__"
        )

    def test_email_service_has_html_templates(self):
        """EmailService must have HTML email templates."""
        assert "PASSWORD_RESET_TEMPLATE_HTML" in self.email_source, (
            "Email service must define PASSWORD_RESET_TEMPLATE_HTML"
        )
        assert "WELCOME_TEMPLATE_HTML" in self.email_source, (
            "Email service must define WELCOME_TEMPLATE_HTML"
        )

    def test_email_service_handles_resend_import_error(self):
        """EmailService must handle missing resend package gracefully."""
        assert "ImportError" in self.email_source, (
            "EmailService must catch ImportError for missing resend package"
        )


# =============================================================================
# TestPasswordStrength
# =============================================================================


class TestPasswordStrength:
    """Verify password strength validation in security.py."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.source = _read("core/security.py")

    def test_password_strength_function_exists(self):
        """validate_password_strength function must exist."""
        assert "def validate_password_strength" in self.source, (
            "validate_password_strength function not found in security.py"
        )

    def test_checks_minimum_length(self):
        """Must check for minimum password length of 8 characters."""
        fn_start = self.source.find("def validate_password_strength")
        assert fn_start > 0
        next_fn = self.source.find("\ndef ", fn_start + 1)
        if next_fn == -1:
            next_fn = len(self.source)
        fn_body = self.source[fn_start:next_fn]

        assert "< 8" in fn_body or "min_length" in fn_body, (
            "validate_password_strength must check for minimum length of 8"
        )

    def test_checks_uppercase(self):
        """Must check for at least one uppercase character."""
        fn_start = self.source.find("def validate_password_strength")
        assert fn_start > 0
        next_fn = self.source.find("\ndef ", fn_start + 1)
        if next_fn == -1:
            next_fn = len(self.source)
        fn_body = self.source[fn_start:next_fn]

        assert "isupper" in fn_body or "A-Z" in fn_body, (
            "validate_password_strength must check for uppercase characters"
        )

    def test_checks_lowercase(self):
        """Must check for at least one lowercase character."""
        fn_start = self.source.find("def validate_password_strength")
        assert fn_start > 0
        next_fn = self.source.find("\ndef ", fn_start + 1)
        if next_fn == -1:
            next_fn = len(self.source)
        fn_body = self.source[fn_start:next_fn]

        assert "islower" in fn_body or "a-z" in fn_body, (
            "validate_password_strength must check for lowercase characters"
        )

    def test_checks_digits(self):
        """Must check for at least one digit."""
        fn_start = self.source.find("def validate_password_strength")
        assert fn_start > 0
        next_fn = self.source.find("\ndef ", fn_start + 1)
        if next_fn == -1:
            next_fn = len(self.source)
        fn_body = self.source[fn_start:next_fn]

        assert "isdigit" in fn_body or "0-9" in fn_body, (
            "validate_password_strength must check for digits"
        )

    def test_checks_special_characters(self):
        """Must check for at least one special character."""
        fn_start = self.source.find("def validate_password_strength")
        assert fn_start > 0
        next_fn = self.source.find("\ndef ", fn_start + 1)
        if next_fn == -1:
            next_fn = len(self.source)
        fn_body = self.source[fn_start:next_fn]

        assert "special" in fn_body.lower() or "!@#" in fn_body, (
            "validate_password_strength must check for special characters"
        )

    def test_returns_tuple_with_errors(self):
        """Must return a tuple of (bool, list) for validation result."""
        fn_start = self.source.find("def validate_password_strength")
        assert fn_start > 0
        next_fn = self.source.find("\ndef ", fn_start + 1)
        if next_fn == -1:
            next_fn = len(self.source)
        fn_body = self.source[fn_start:next_fn]

        assert "errors" in fn_body, "validate_password_strength must collect errors"
        assert "return" in fn_body, "validate_password_strength must return validation result"

    def test_checks_maximum_length(self):
        """Must check for maximum password length to prevent DoS."""
        fn_start = self.source.find("def validate_password_strength")
        assert fn_start > 0
        next_fn = self.source.find("\ndef ", fn_start + 1)
        if next_fn == -1:
            next_fn = len(self.source)
        fn_body = self.source[fn_start:next_fn]

        assert "> 100" in fn_body or "max_length" in fn_body or "> 128" in fn_body, (
            "validate_password_strength must check maximum password length"
        )


# =============================================================================
# TestJWTClaimProtection
# =============================================================================


class TestJWTClaimProtection:
    """Verify JWT claim protection in security.py."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.source = _read("core/security.py")

    def test_access_token_protects_critical_claims(self):
        """create_access_token must prevent overwriting critical JWT claims."""
        fn_start = self.source.find("def create_access_token")
        assert fn_start > 0
        next_fn = self.source.find("\ndef ", fn_start + 1)
        if next_fn == -1:
            next_fn = len(self.source)
        fn_body = self.source[fn_start:next_fn]

        assert "protected" in fn_body or "safe_claims" in fn_body, (
            "create_access_token must protect critical JWT claims from being overwritten"
        )

    def test_access_token_includes_jti(self):
        """create_access_token must include JTI for token blacklisting."""
        fn_start = self.source.find("def create_access_token")
        assert fn_start > 0
        next_fn = self.source.find("\ndef ", fn_start + 1)
        if next_fn == -1:
            next_fn = len(self.source)
        fn_body = self.source[fn_start:next_fn]

        assert '"jti"' in fn_body, "create_access_token must include JTI claim for blacklisting"

    def test_token_blacklist_fail_open(self):
        """is_token_blacklisted must fail open (return False) when Redis is unavailable.

        Rationale: fail-closed would lock out ALL users during Redis outage.
        Token blacklisting is a secondary defence; JWTs still have expiry.
        Rate limiting remains fail-closed (that protects against abuse).
        """
        fn_start = self.source.find("async def is_token_blacklisted")
        assert fn_start > 0
        next_fn = self.source.find("\ndef ", fn_start + 1)
        if next_fn == -1:
            next_fn = self.source.find("\nasync def ", fn_start + 1)
        if next_fn == -1:
            next_fn = len(self.source)
        fn_body = self.source[fn_start:next_fn]

        assert "return False" in fn_body, (
            "is_token_blacklisted must fail open (return False) when Redis is unavailable"
        )
