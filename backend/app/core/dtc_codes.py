"""
DTC (Diagnostic Trouble Code) rules - single source of truth.

Every layer that has to decide "is this string a diagnostic trouble code?"
must use this module: the API request validators, the metrics endpoint
normalizer, and the standalone import/CLI scripts under ``scripts/``.
Before this module existed the project carried ten independent regexes that
disagreed with each other, and two of them were actively causing bugs.

SAE J2012 / OBD-II structure of a diagnostic trouble code::

    char 1   : P (powertrain) | B (body) | C (chassis) | U (network)
    char 2   : 0-3   (0/2 = SAE generic, 1/3 = manufacturer specific)
    char 3-5 : hex   (0-9 A-F) subsystem + fault index

Why the second character MUST be constrained to ``0-3``
-------------------------------------------------------
That single character is the whole precision story, and it is exactly what
the two historical patterns got wrong - in opposite directions:

``[PBCU][0-9]{4}`` (too strict - four DECIMAL digits)
    Rejects every hex DTC that real scan tools return: ``P26B7``, ``P090C``,
    ``P0A94``, ``B00A0``, ``P17F1``, ``P324E``. In the CLI and importer
    scripts this silently DROPPED those codes from the imported corpus; in
    ``/api/v1/diagnosis/quick-analyze`` the equivalent ``code[1:].isdigit()``
    check answers a legitimate code typed by a mechanic with HTTP 400. It
    also still ACCEPTS ``P9324``, a Nissan *service campaign* number,
    because it never checked char 2.

``[PBCU][0-9A-F]{4}`` (too loose - any four hex-ish characters)
    P/B/C/U followed by hex letters spells ordinary English: ``PEACE``,
    ``PACED``, ``BEEDA`` all match. This is how ``PEACE``, ``PACED``,
    ``P93AF``, ``UA80E`` and ``UA80F`` were imported into
    ``data/dtc_codes/all_codes_complete.json`` as if they were DTCs, and how
    Toyota transmission designations (``U760E``) and Nissan campaign IDs
    (``PC861``, ``PC214``, ``PC490``) polluted the graph. This is the variant
    that sat in the ``DiagnosisRequest`` / ``InspectionRequest`` validators,
    so every one of those junk strings passed request validation and flowed
    on into Neo4j lookups, RAG cache keys and log lines.

Requiring ``0-3`` kills both failure modes at once: ``PEACE``/``PACED``
(char 2 = ``E``/``A``), ``UA80E``/``UA80F`` (``A``), ``P93AF``/``P9324``
(``9``), ``U760E`` (``7``), ``PC861`` (``C``) are all rejected, while every
real hex code is kept because only characters 3-5 are hex.

Measured on 26,237 real NHTSA complaint narratives (30 make/model/year sets
pulled from api.nhtsa.gov)::

    old strict : 373 mentions, of which 15 false positives ("P9324")
    old loose  : 478 mentions, of which 39 false positives (PEACE, PC861,
                 PC214, PC490, PC426, PC491, U760E, BEEDA, P9324)
    this one   : 443 mentions, 0 observed false positives
                 (+19% recall vs. strict, -7% volume vs. loose = the junk)

Two entry points, deliberately kept separate
--------------------------------------------
* :func:`is_valid_dtc_code` / :func:`normalize_dtc_code` - validate ONE
  user-supplied code. Anchored: the whole string must be a code.
* :func:`extract_dtc_codes` / :func:`contains_dtc_code` - mine free text for
  codes. Boundary-guarded so a code is never matched inside a longer token
  (VIN fragments such as ``1FADP3F25FL``, part numbers, ``XP0301``), and
  tolerant of the separators humans type (``P-0301``, ``P 0301``).

This module is intentionally dependency-free (stdlib ``re`` only) so the
standalone scripts can load it without booting the FastAPI settings object.

The canonical reference implementation these semantics mirror lives in
``scripts/sync_neo4j_sprint9.py``; ``backend/tests/unit/test_dtc_codes.py``
contains a drift guard asserting the two never diverge.
"""

import re
from typing import Dict, List, Optional

__all__ = [
    "DTC_CATEGORY_BY_PREFIX",
    "DTC_CODE_PATTERN",
    "DTC_CODE_STRICT",
    "contains_dtc_code",
    "dtc_category",
    "extract_dtc_codes",
    "is_valid_dtc_code",
    "normalize_dtc_code",
]

# Free-text extraction. The lookarounds (rather than \b) are what stop a code
# from being matched inside a longer alphanumeric token: \b would happily find
# "P3F25" inside the VIN fragment "1FADP3F25FL".
DTC_CODE_PATTERN = re.compile(
    r"(?<![0-9A-Za-z])"  # left boundary - never match inside a longer token
    r"([PBCUpbcu])"  # system letter
    r"[\s\-]?"  # optional single separator: "P-0301", "P 0301"
    r"([0-3][0-9A-Fa-f]{3})"  # 0-3 + 3 hex digits
    r"(?![0-9A-Za-z])"  # right boundary
)

# Canonical shape of a single, already-normalised code. Used to validate one
# user-supplied string (API request bodies, path segments, CLI arguments).
DTC_CODE_STRICT = re.compile(r"^[PBCU][0-3][0-9A-F]{3}$")

DTC_CATEGORY_BY_PREFIX: Dict[str, str] = {
    "P": "powertrain",
    "B": "body",
    "C": "chassis",
    "U": "network",
}


def is_valid_dtc_code(code: Optional[str]) -> bool:
    """True if `code` is a structurally valid OBD-II DTC (after upper-casing).

    Surrounding whitespace is tolerated, so `" p0300 "` validates. Anything
    else - wrong length, a non-P/B/C/U first character, a second character
    outside 0-3, or a non-hex tail - is rejected.
    """
    if not code:
        return False
    return bool(DTC_CODE_STRICT.match(code.strip().upper()))


def normalize_dtc_code(code: Optional[str]) -> Optional[str]:
    """Return the canonical uppercase form of `code`, or None if invalid.

    The validate-and-canonicalise entry point for API boundaries: every
    downstream consumer (Neo4j graph lookup, RAG cache key, log line) then
    receives the same spelling, so `"p0300 "` and `"P0300"` cannot produce a
    cache miss or a failed graph lookup.
    """
    if not code:
        return None
    canonical = code.strip().upper()
    return canonical if is_valid_dtc_code(canonical) else None


def contains_dtc_code(text: Optional[str]) -> bool:
    """True if `text` mentions at least one DTC code.

    Equivalent to `bool(extract_dtc_codes(text))` but short-circuits on the
    first match, which matters on corpus-sized scans.
    """
    if not text:
        return False
    return DTC_CODE_PATTERN.search(text) is not None


def extract_dtc_codes(text: Optional[str]) -> List[str]:
    """Extract structurally valid DTC codes from free text.

    Case-insensitive, tolerates a single space or hyphen between the system
    letter and the digits, and never matches inside a longer alphanumeric
    token (so VIN fragments and part numbers are rejected).

    Returns a sorted list of unique upper-case codes.
    """
    if not text:
        return []
    return sorted(
        {f"{letter.upper()}{digits.upper()}" for letter, digits in DTC_CODE_PATTERN.findall(text)}
    )


def dtc_category(code: str) -> str:
    """Category for a DTC code prefix ("powertrain", "body", ...)."""
    return DTC_CATEGORY_BY_PREFIX.get(code[:1].upper(), "unknown")
