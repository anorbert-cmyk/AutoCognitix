"""
PII redaction helpers shared across logging and error reporting.

UUIDs are correlatable to user_id and VINs identify natural persons (GDPR),
so both must be stripped from any text that leaves the system (e.g. Sentry
events). The patterns mirror the ones used by
``app.middleware.metrics.EndpointNormalizer`` (UUID_PATTERN / VIN_PATTERN),
which normalizes the same identifiers for Prometheus label cardinality.
"""

import re

# UUID (any version) — case-insensitive hex groups 8-4-4-4-12.
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)

# VIN: exactly 17 alphanumeric chars excluding I, O, Q (ISO 3779).
# Case-insensitive: VINs arrive lowercase in URLs as well.
_VIN_RE = re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b", re.IGNORECASE)


def redact_pii(text: str) -> str:
    """Replace UUIDs and VINs in *text* with neutral placeholders.

    UUIDs become ``<uuid>`` and VINs become ``<vin>``. UUIDs are substituted
    first; their hyphenated groups can never satisfy the 17-char VIN pattern,
    so the order only matters for determinism.
    """
    text = _UUID_RE.sub("<uuid>", text)
    text = _VIN_RE.sub("<vin>", text)
    return text
