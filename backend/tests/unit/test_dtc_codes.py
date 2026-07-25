"""
Unit tests for the shared DTC primitive in ``app.core.dtc_codes``.

This module is the single source of truth for "is this string a diagnostic
trouble code?" across the API request validators, the metrics endpoint
normalizer and the standalone scripts. The tests cover both entry points:

  - the single-code validator (``is_valid_dtc_code`` / ``normalize_dtc_code``),
  - the free-text extractor (``extract_dtc_codes`` / ``contains_dtc_code``),

plus drift guards asserting that the reference implementation in
``scripts/sync_neo4j_sprint9.py`` and every migrated script agree with it.

No database, no network: the scripts are loaded by path with importlib.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from app.core.dtc_codes import (
    DTC_CODE_PATTERN,
    DTC_CODE_STRICT,
    contains_dtc_code,
    dtc_category,
    extract_dtc_codes,
    is_valid_dtc_code,
    normalize_dtc_code,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def _load_script(relative_path: str) -> ModuleType:
    """Import a standalone script from scripts/ without installing it."""
    path = SCRIPTS_DIR / relative_path
    if not path.exists():  # pragma: no cover - repo layout guard
        pytest.skip(f"script not found: {path}")
    name = f"{path.stem}_under_test_dtc_codes"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Validator: real codes must be accepted
# ---------------------------------------------------------------------------
REAL_CODES = [
    # decimal codes - accepted by every historical pattern
    "P0300",
    "P0301",
    "P0171",
    "P0420",
    "U0100",
    "B0101",
    "C1391",
    # hex codes - silently DROPPED by the old strict `[PBCU][0-9]{4}` used in
    # the CLI/importer scripts, so real scan-tool readings never reached the DB
    "P26B7",
    "P090C",
    "P0A94",
    "B00A0",
    "P17F1",
    "U3000",
    "C3FFF",
    # manufacturer-specific second characters 1..3
    "P1450",
    "P324E",
    "B1234",
    "U1213",
]


@pytest.mark.unit
@pytest.mark.parametrize("code", REAL_CODES)
def test_is_valid_dtc_code_accepts_real_codes(code):
    assert is_valid_dtc_code(code) is True


@pytest.mark.unit
@pytest.mark.parametrize("code", REAL_CODES)
def test_is_valid_dtc_code_is_case_insensitive(code):
    assert is_valid_dtc_code(code.lower()) is True
    assert is_valid_dtc_code(code.swapcase()) is True


@pytest.mark.unit
@pytest.mark.parametrize("raw", [" P0301 ", "\tp26b7\n", "p0a94", "P0A94"])
def test_normalize_dtc_code_canonicalises(raw):
    canonical = normalize_dtc_code(raw)
    assert canonical == raw.strip().upper()
    assert is_valid_dtc_code(canonical) is True


# ---------------------------------------------------------------------------
# Validator: the documented false positives must be rejected
# ---------------------------------------------------------------------------
KNOWN_FALSE_POSITIVES = [
    # English words matched by the old hex-permissive pattern
    "PEACE",
    "PACED",
    "BEEDA",
    # values that actually reached data/dtc_codes/all_codes_complete.json
    "UA80F",
    "UA80E",
    "P93AF",
    # Nissan service campaign IDs / Toyota transmission designations found in
    # real NHTSA narratives (P9324 was matched even by the OLD strict pattern)
    "P9324",
    "PC861",
    "PC214",
    "PC490",
    "U760E",
]


@pytest.mark.unit
@pytest.mark.parametrize("token", KNOWN_FALSE_POSITIVES)
def test_is_valid_dtc_code_rejects_known_false_positives(token):
    assert is_valid_dtc_code(token) is False
    assert is_valid_dtc_code(token.lower()) is False
    assert normalize_dtc_code(token) is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "code",
    [
        None,
        "",
        "   ",
        "P030",  # too short
        "P03011",  # too long
        "X0301",  # not a P/B/C/U system letter
        "0301",  # no system letter
        "P0OO1",  # O is not a hex digit
        "PABCD",  # second character not 0-3
        "P-0301",  # separators are an extraction affordance, not a valid code
        "P 0301",
        "P0301,P0302",  # a list is not a single code
    ],
)
def test_is_valid_dtc_code_rejects_junk(code):
    assert is_valid_dtc_code(code) is False
    assert normalize_dtc_code(code) is None


# ---------------------------------------------------------------------------
# Extractor: recall
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("P0301", ["P0301"]),
        ("P26B7", ["P26B7"]),
        ("P090C", ["P090C"]),
        ("P0A94", ["P0A94"]),
        ("B00A0", ["B00A0"]),
        ("p0171", ["P0171"]),
        ("Dealer found code p26b7 stored", ["P26B7"]),
        ("MIL ON, DTC: P0301", ["P0301"]),
        # separator forms the reference implementation supports
        ("code P-0301 was pulled", ["P0301"]),
        ("scanner showed P 0300 misfire", ["P0300"]),
        ("codes p0301, P0302 and p0303", ["P0301", "P0302", "P0303"]),
        # punctuation boundaries
        ("(P0301)", ["P0301"]),
        ("P0301.", ["P0301"]),
        ("P0301,P0302", ["P0301", "P0302"]),
        ("codes: P0301/P0420", ["P0301", "P0420"]),
        ("P0301-P0304", ["P0301", "P0304"]),
    ],
)
def test_extract_dtc_codes_recall(text, expected):
    assert extract_dtc_codes(text) == expected


@pytest.mark.unit
def test_extract_dtc_codes_realistic_narrative():
    text = (
        "THE CONTACT OWNS A 2015 FORD FOCUS. THE CHECK ENGINE LIGHT ILLUMINATED "
        "AND THE DEALER DIAGNOSED CODES P0301 AND p0304 (MISFIRE), PLUS P26B7."
    )
    assert extract_dtc_codes(text) == ["P0301", "P0304", "P26B7"]


# ---------------------------------------------------------------------------
# Extractor: precision
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize("token", KNOWN_FALSE_POSITIVES)
def test_extract_dtc_codes_rejects_known_false_positives(token):
    assert extract_dtc_codes(token) == []
    assert extract_dtc_codes(f"the {token} appeared in the report") == []
    assert extract_dtc_codes(token.lower()) == []


@pytest.mark.unit
def test_extract_dtc_codes_ignores_campaign_ids_in_narrative():
    text = (
        "It has been reported on Nissan Service Campaign P9324, dated October 25, "
        "2019. The dealer also mentioned campaign PC861 for the headlamps."
    )
    assert extract_dtc_codes(text) == []


@pytest.mark.unit
@pytest.mark.parametrize(
    "text",
    [
        "1FADP3F25FL",  # VIN fragment contains P3F25
        "XP0301",
        "P0301A",
        "P03011",
        "AP0300Z",
        "part number BP0171X",
    ],
)
def test_extract_dtc_codes_ignores_codes_inside_longer_tokens(text):
    assert extract_dtc_codes(text) == []


@pytest.mark.unit
@pytest.mark.parametrize("text", [None, "", "   ", "no codes here at all"])
def test_extract_dtc_codes_empty_and_codeless(text):
    assert extract_dtc_codes(text) == []


@pytest.mark.unit
def test_extract_dtc_codes_dedups_and_orders_deterministically():
    text = "P0302 then P0301 then p0302 again, plus P0A94 and p0a94"
    result = extract_dtc_codes(text)
    assert result == ["P0301", "P0302", "P0A94"]
    assert result == sorted(set(result))
    # stable across calls and independent of mention order
    assert extract_dtc_codes("p0a94 P0302 P0301 P0302") == result


# ---------------------------------------------------------------------------
# contains_dtc_code must be the short-circuiting twin of extract_dtc_codes
# ---------------------------------------------------------------------------
CONTAINS_CORPUS = [
    None,
    "",
    "   ",
    "P0301",
    "p26b7 stored",
    "PEACE of mind",
    "Nissan Service Campaign P9324",
    "VIN 1FADP3F25FL",
    "no codes here at all",
    "error code B00A0 present",
    "codes: P0301/P0420",
]


@pytest.mark.unit
@pytest.mark.parametrize("text", CONTAINS_CORPUS)
def test_contains_dtc_code_matches_extractor(text):
    assert contains_dtc_code(text) == bool(extract_dtc_codes(text))


# ---------------------------------------------------------------------------
# Category helper
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("P0301", "powertrain"),
        ("b0101", "body"),
        ("C1391", "chassis"),
        ("U0100", "network"),
        ("X0301", "unknown"),
        ("", "unknown"),
    ],
)
def test_dtc_category(code, expected):
    assert dtc_category(code) == expected


# ---------------------------------------------------------------------------
# Drift guard: the shared primitive vs. the reference implementation
# ---------------------------------------------------------------------------
# A single corpus exercised by both implementations. Any divergence in the
# regex, the boundary handling, the case folding or the ordering fails here.
DRIFT_CORPUS = [
    *REAL_CODES,
    *KNOWN_FALSE_POSITIVES,
    *[c.lower() for c in REAL_CODES],
    None,
    "",
    "   ",
    " P0301 ",
    "P030",
    "P03011",
    "X0301",
    "P0OO1",
    "PABCD",
    "P0301 and P0302",
    "DTC: p26b7 stored",
    "trouble code P0A94",
    "code P-0301 was pulled",
    "scanner showed P 0300 misfire",
    "PEACE of mind",
    "Nissan Service Campaign P9324",
    "transmission U760E shudder",
    "VIN 1FADP3F25FL",
    "XP0301",
    "P0301A",
    "AP0300Z",
    "part number BP0171X",
    "error code B00A0 present",
    "codes: P0301/P0420",
    "P0301-P0304",
    "P0302 then P0301 then p0302 again",
    "no codes here at all",
]


@pytest.mark.unit
def test_no_drift_from_reference_implementation():
    """`app.core.dtc_codes` must agree with `scripts/sync_neo4j_sprint9.py`.

    That script is the audited reference (85 tests in test_dtc_extraction.py)
    and cannot import the backend package, so the two definitions can only be
    kept honest by asserting they behave identically on a shared corpus.
    """
    pytest.importorskip("neo4j", reason="neo4j driver required to import the sync script")
    reference = _load_script("sync_neo4j_sprint9.py")

    assert reference.DTC_CODE_PATTERN.pattern == DTC_CODE_PATTERN.pattern
    assert reference.DTC_CODE_STRICT.pattern == DTC_CODE_STRICT.pattern

    for text in DRIFT_CORPUS:
        assert extract_dtc_codes(text) == reference.extract_dtc_codes(text), text
        assert is_valid_dtc_code(text) == reference.is_valid_dtc_code(text), text

    for code in [*REAL_CODES, "X0301", ""]:
        assert dtc_category(code) == reference.dtc_category(code), code


# ---------------------------------------------------------------------------
# Drift guard: the migrated scripts must delegate, not re-implement
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_sample_complaints_uses_the_shared_primitive():
    module = _load_script("sample_complaints.py")
    for text in DRIFT_CORPUS:
        expected = contains_dtc_code(text)
        assert module.has_dtc_code({"summary": text}) is expected, text
    assert module.has_dtc_code({}) is False


@pytest.mark.unit
def test_validate_data_uses_the_shared_primitive():
    module = _load_script("validate_data.py")
    for text in DRIFT_CORPUS:
        if text is None:
            continue
        assert module.validate_dtc_format(text) == is_valid_dtc_code(text), text
    for code in [*REAL_CODES, "X0301", ""]:
        assert module.get_category_from_code(code) == dtc_category(code), code


@pytest.mark.unit
def test_sync_postgres_uses_the_shared_primitive():
    pytest.importorskip("psycopg2", reason="psycopg2 required to import sync_postgres_sprint9")
    pytest.importorskip("tqdm", reason="tqdm required to import sync_postgres_sprint9")
    module = _load_script("sync_postgres_sprint9.py")
    for text in DRIFT_CORPUS:
        assert module.extract_dtc_codes(text) == extract_dtc_codes(text), text


@pytest.mark.unit
@pytest.mark.parametrize(
    ("relative_path", "dependency"),
    [
        ("diagnose.py", "typer"),
        ("cli/diagtool.py", "typer"),
        ("cli/autocognitix_cli.py", "click"),
    ],
)
def test_cli_scripts_use_the_shared_primitive(relative_path, dependency):
    pytest.importorskip(dependency, reason=f"{dependency} required to import {relative_path}")
    module = _load_script(relative_path)
    for text in DRIFT_CORPUS:
        if text is None:
            continue
        assert module.validate_dtc_code(text) == is_valid_dtc_code(text), text


# ---------------------------------------------------------------------------
# API request validation
# ---------------------------------------------------------------------------
# These two schemas previously carried the TOO LOOSE pattern (`[0-9A-F]{4}`),
# so real hex codes already passed - the regression risk of tightening to the
# SAE rule is that they stop passing. That is what the "accepts" cases pin
# down; the "rejects" cases pin down the junk that used to get through.
@pytest.mark.unit
@pytest.mark.parametrize("code", ["P26B7", "P090C", "P0A94", "B00A0", "p26b7", " P0A94 "])
def test_diagnosis_request_accepts_real_hex_dtc_codes(code):
    """Real manufacturer/hex codes must keep passing request validation."""
    from app.api.v1.schemas.diagnosis import DiagnosisRequest

    request = DiagnosisRequest(
        vehicle_make="Volkswagen",
        vehicle_model="Golf",
        vehicle_year=2018,
        dtc_codes=[code],
        symptoms="A motor rangat es a check engine lampa vilagit.",
    )
    assert request.dtc_codes == [code.strip().upper()]


@pytest.mark.unit
@pytest.mark.parametrize("code", ["PEACE", "UA80E", "P93AF", "P9324", "INVALID", "X0300"])
def test_diagnosis_request_still_rejects_junk(code):
    from pydantic import ValidationError

    from app.api.v1.schemas.diagnosis import DiagnosisRequest

    with pytest.raises(ValidationError):
        DiagnosisRequest(
            vehicle_make="Volkswagen",
            vehicle_model="Golf",
            vehicle_year=2018,
            dtc_codes=[code],
            symptoms="A motor rangat es a check engine lampa vilagit.",
        )


@pytest.mark.unit
@pytest.mark.parametrize("code", ["P26B7", "P090C", "P0A94", "B00A0", "p26b7"])
def test_inspection_request_accepts_real_hex_dtc_codes(code):
    """Real manufacturer/hex codes must keep passing request validation."""
    from app.api.v1.schemas.inspection import InspectionRequest

    request = InspectionRequest(
        vehicle_make="Volkswagen",
        vehicle_model="Golf",
        vehicle_year=2018,
        dtc_codes=[code],
    )
    assert request.dtc_codes == [code.upper()]


@pytest.mark.unit
@pytest.mark.parametrize("code", ["PEACE", "UA80E", "P93AF", "P9324", "INVALID", "X0300"])
def test_inspection_request_still_rejects_junk(code):
    from pydantic import ValidationError

    from app.api.v1.schemas.inspection import InspectionRequest

    with pytest.raises(ValidationError):
        InspectionRequest(
            vehicle_make="Volkswagen",
            vehicle_model="Golf",
            vehicle_year=2018,
            dtc_codes=[code],
        )


# ---------------------------------------------------------------------------
# The metrics endpoint normalizer shares the same rule
# ---------------------------------------------------------------------------
# `app.middleware.metrics` declares Info("autocognitix_app_info"), which
# collides on the default Prometheus registry with Info("autocognitix_app") in
# `app.core.metrics`. That pre-existing clash means the two modules can never
# be imported into the same interpreter, so this case runs in a fresh one.
_NORMALIZER_PROBE = """
import sys
sys.path.insert(0, {backend!r})
from app.middleware.metrics import EndpointNormalizer as N

for code in ["P0300", "P26B7", "p0a94", "B00A0", "P0301"]:
    got = N.normalize("/api/v1/dtc/" + code)
    assert got == "/api/v1/dtc/{{dtc_code}}", (code, got)

for junk in ["PEACE", "UA80E", "P93AF", "P9324", "search", "stats"]:
    got = N.normalize("/api/v1/dtc/" + junk)
    assert got == "/api/v1/dtc/" + junk, (junk, got)

# other segment kinds must keep working
assert N.normalize("/api/v1/vehicles/1HGBH41JXMN109186") == "/api/v1/vehicles/{{vin}}"
assert N.normalize("/api/v1/items/12345") == "/api/v1/items/{{id}}"
assert N.normalize("/health") == "/health"
print("OK")
"""


@pytest.mark.unit
def test_endpoint_normalizer_uses_the_shared_dtc_rule():
    backend_dir = str(Path(__file__).resolve().parents[2])
    result = subprocess.run(
        [sys.executable, "-c", _NORMALIZER_PROBE.format(backend=backend_dir)],
        capture_output=True,
        text=True,
        cwd=backend_dir,
        env=os.environ.copy(),
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("OK")
