"""
Unit tests for the shared DTC primitive in ``app.core.dtc_codes``.

This module is the single source of truth for "is this string a diagnostic
trouble code?" across the API request validators, the metrics endpoint
normalizer and the standalone scripts. The tests cover both entry points:

  - the single-code validator (``is_valid_dtc_code`` / ``normalize_dtc_code``),
  - the free-text extractor (``extract_dtc_codes`` / ``contains_dtc_code``),

plus a two-layer drift guard over ``scripts/``:

  - layer 1 (always runs, source-level): no file under ``scripts/`` may carry a
    DTC regex of its own, and every DTC-aware script must import the canonical
    module. Uses ``ast``/``tokenize``, so it needs none of the scripts'
    runtime dependencies.
  - layer 2 (behavioural): for the scripts that import cleanly, the wrapper
    around the canonical function must give the same answers.

No database, no network: the scripts are loaded by path with importlib.
"""

import ast
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import tokenize
from pathlib import Path
from types import ModuleType

import pytest

from app.core.dtc_codes import (
    contains_dtc_code,
    dtc_category,
    extract_dtc_codes,
    is_valid_dtc_code,
    normalize_dtc_code,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
BACKEND_DIR = PROJECT_ROOT / "backend"


def _subprocess_env(**overrides: str) -> dict:
    """Environment for a probe subprocess, with coverage tracing switched off.

    pytest-cov activates itself in child processes through a .pth file driven by
    these variables. A child that does not run from the repo root cannot find
    pyproject.toml, so it records STATEMENT coverage while the parent records
    BRANCH coverage (``branch = true``), and the run dies in teardown with
    ``DataError: Can't combine statement coverage data with branch data`` -
    after every test has passed. These probes assert behaviour, not coverage,
    so they simply opt out.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith(("COV_CORE_", "COVERAGE_"))}
    env.update(overrides)
    return env


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


# ---------------------------------------------------------------------------
# Drift guard, layer 1 (always runs): nothing may re-implement the rule
# ---------------------------------------------------------------------------
# The old sync_postgres guard was gated on `pytest.importorskip("psycopg2")`.
# psycopg2 is not in backend/requirements.txt, which is the only install step
# in .github/workflows/ci.yml, so that guard was PERMANENTLY skipped and the
# copy it was meant to protect could drift with nothing failing.
#
# The guards below read SOURCE instead of importing it, so they always run: no
# database driver, no optional dependency, no network. They are also strictly
# stronger than what they replace, because the first one polices every file in
# scripts/ rather than the handful somebody remembered to list.
#
# Layer 2 (further down) keeps the behavioural comparisons for the scripts that
# import cleanly, so wrapper functions are checked as well as import lines.

# A P/B/C/U character class immediately followed by a four-wide quantifier.
# That is precisely the shape of both historical mistakes - `[PBCU][0-9]{4}`
# (too strict, drops every hex code) and `[PCBU][0-9A-Fa-f]{4}` (too loose,
# imports PEACE / PACED / U760E / PC861). The canonical
# `[PBCU][0-3][0-9A-F]{3}` does not match it, because the second character
# class is followed by another class rather than by `{4}`.
FORBIDDEN_DTC_REGEX_SHAPE = re.compile(r"\[[PBCUpbcu]{4,8}\]\s*(?:\[[^\]]+\]|\\d)\{4\}")

# Two survivors that deliberately keep a decimal four-digit shape because they
# do NOT answer "is this string a DTC?". Listed explicitly (file -> reason) so
# a NEW copy can never hide behind a blanket exclusion.
FORBIDDEN_SHAPE_ALLOWLIST = {
    # Parses section headers of the dtcdb CSV ("DTC Codes - P0100-P0199 - ...").
    # The range endpoints are that file's own decimal section labels, so the
    # count is 2: one per end of the range.
    "import_dtcdb.py": 2,
    # Security allowlist for a URL path segment ("p0000-p0099") used to build
    # scraper URLs; it guards against injection, it does not classify codes.
    # Also a range, hence 2.
    "utils/url_validator.py": 2,
}


def _strip_comments(source: str) -> str:
    """Return `source` with comment tokens removed (docstrings are kept)."""
    out = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.COMMENT:
            out.append(token.string)
    return "\n".join(out)


def _script_paths():
    return sorted(p for p in SCRIPTS_DIR.rglob("*.py") if "__pycache__" not in p.parts)


@pytest.mark.unit
def test_no_script_reimplements_the_dtc_rule():
    """No file under scripts/ may carry a DTC regex of its own.

    This is the guard that would have caught the three point-fixes: it fails on
    a NEW copy appearing anywhere in scripts/, not just in a listed file.
    """
    offenders = {}
    for path in _script_paths():
        relative = path.relative_to(SCRIPTS_DIR).as_posix()
        hits = FORBIDDEN_DTC_REGEX_SHAPE.findall(_strip_comments(path.read_text(encoding="utf-8")))
        allowed = FORBIDDEN_SHAPE_ALLOWLIST.get(relative, 0)
        if len(hits) > allowed:
            offenders[relative] = hits

    assert not offenders, (
        "These files re-implement the DTC rule instead of importing "
        f"app.core.dtc_codes: {offenders}"
    )


# Every script that decides "is this string a DTC?". Each must reach the
# canonical module - directly, or through scripts/utils.py (asserted below to
# be canonical itself).
DTC_AWARE_SCRIPTS = [
    "cli/autocognitix_cli.py",
    "cli/diagtool.py",
    "diagnose.py",
    "download_all_obdb.py",
    "import_data.py",
    "import_dtcdb.py",
    "import_obd_codes.py",
    "import_obdb.py",
    "import_obdb_github.py",
    "import_training_data.py",
    "load_all_to_neo4j.py",
    "merge_dtc_all_sources.py",
    "merge_dtc_sources.py",
    "sample_complaints.py",
    "scrape_autocodes.py",
    "scrape_bbareman.py",
    "scrape_dtcbase.py",
    "scrape_engine_codes.py",
    "scrape_klavkarr.py",
    "scrape_obd_codes.py",
    "scrape_troublecodes.py",
    "sync_neo4j_sprint9.py",
    "sync_nhtsa.py",
    "sync_nhtsa_complete.py",
    "sync_nhtsa_vehicles.py",
    "sync_postgres_sprint9.py",
    "utils.py",
    "validate_data.py",
]

CANONICAL_MODULES = {"app.core.dtc_codes", "backend.app.core.dtc_codes"}
# scripts/utils.py re-exports the canonical validator; importing from it counts.
SHARED_UTILS_MODULES = {"scripts.utils", "utils"}
SHARED_UTILS_NAMES = {"validate_dtc_code", "get_category_from_code"}


def _reaches_canonical_rules(path):
    """True if the module imports the canonical DTC rules (AST, no execution)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        names = {alias.name for alias in node.names}
        # from app.core.dtc_codes import ... / from backend.app.core.dtc_codes import ...
        if node.module in CANONICAL_MODULES:
            return True
        # from app.core import dtc_codes [as dtc_rules]
        if node.module.endswith("app.core") and "dtc_codes" in names:
            return True
        # from scripts.utils import validate_dtc_code
        if node.module in SHARED_UTILS_MODULES and names & SHARED_UTILS_NAMES:
            return True
    return False


@pytest.mark.unit
@pytest.mark.parametrize("relative_path", DTC_AWARE_SCRIPTS)
def test_script_imports_the_canonical_rules(relative_path):
    """Each DTC-aware script must IMPORT the rule, never restate it.

    Source-level (ast.parse), so this runs without neo4j, psycopg2, typer,
    click, httpx or bs4 installed - unlike the guards it replaced.
    """
    path = SCRIPTS_DIR / relative_path
    assert path.exists(), f"script listed in DTC_AWARE_SCRIPTS is missing: {path}"
    assert _reaches_canonical_rules(path), (
        f"{relative_path} does not import app.core.dtc_codes (directly or via scripts/utils.py)"
    )


@pytest.mark.unit
def test_shared_scripts_utils_delegates_to_the_canonical_rules():
    """scripts/utils.py is the gateway for six scrapers - it must delegate."""
    assert _reaches_canonical_rules(SCRIPTS_DIR / "utils.py")


@pytest.mark.unit
def test_canonical_module_is_importable_without_settings():
    """`app.core.dtc_codes` must import with no SECRET_KEY and no .env.

    This is the property that lets the standalone scripts import the rule
    instead of copying it. It broke silently before, because
    `app/core/__init__.py` eagerly imported `app.core.config`, which builds the
    Pydantic Settings object. Run in a subprocess with a scrubbed environment
    and a cwd that holds no .env, so an ambient secret cannot mask a
    regression.
    """
    env = {
        k: v
        for k, v in _subprocess_env().items()
        if k not in {"SECRET_KEY", "JWT_SECRET_KEY"} and not k.startswith("AUTOCOGNITIX_")
    }
    env["PYTHONPATH"] = str(BACKEND_DIR)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            # One deliberate multi-line program, not a list of arguments: written
            # as a single triple-quoted string so it cannot be misread (by a
            # human or a linter) as list entries missing a comma.
            """
import sys
from app.core.dtc_codes import is_valid_dtc_code, extract_dtc_codes

assert is_valid_dtc_code('P26B7')
assert extract_dtc_codes('VIN 1FADP3F25FL code P0301') == ['P0301']
assert 'app.core.config' not in sys.modules, 'settings were built on import'
print('OK')
""",
        ],
        cwd=tempfile.gettempdir(),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "OK" in result.stdout


# ---------------------------------------------------------------------------
# Drift guard, layer 2: the scripts that import cleanly must BEHAVE identically
# ---------------------------------------------------------------------------
# Layer 1 proves the import line exists; these prove the wrapper around it did
# not distort the answer (case folding, ordering, set-vs-list, empty input).
@pytest.mark.unit
@pytest.mark.parametrize(
    ("relative_path", "attribute", "dependency"),
    [
        ("sync_neo4j_sprint9.py", "extract_dtc_codes", "neo4j"),
        ("sync_neo4j_sprint9.py", "is_valid_dtc_code", "neo4j"),
        ("sync_neo4j_sprint9.py", "dtc_category", "neo4j"),
    ],
)
def test_sync_neo4j_uses_the_canonical_function_object(relative_path, attribute, dependency):
    """The strongest possible no-drift statement: same function object.

    This replaces the old corpus comparison against
    `scripts/sync_neo4j_sprint9.py`, which was described as "the canonical
    reference implementation". It is not one any more - it imports the rule.
    """
    pytest.importorskip(dependency, reason=f"{dependency} required to import {relative_path}")
    module = _load_script(relative_path)
    assert getattr(module, attribute) is globals()[attribute]


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
def test_scripts_utils_validator_matches_the_canonical_rule():
    """scripts/utils.py feeds six scrapers - check behaviour, not just imports."""
    module = _load_script("utils.py")
    for text in DRIFT_CORPUS:
        if text is None:
            continue
        assert module.validate_dtc_code(text) == is_valid_dtc_code(text), text
    for code in [*REAL_CODES, "X0301", ""]:
        assert module.get_category_from_code(code) == dtc_category(code), code


@pytest.mark.unit
def test_sync_nhtsa_extraction_matches_the_canonical_rule():
    pytest.importorskip("httpx", reason="httpx required to import sync_nhtsa")
    module = _load_script("sync_nhtsa.py")
    for text in DRIFT_CORPUS:
        if text is None:
            continue
        assert module.extract_dtc_codes(text) == set(extract_dtc_codes(text)), text


# These three need a CLI toolkit to import. typer (and therefore click) IS in
# backend/requirements.txt, so they run in CI; the unconditional structural
# guard above is what carries the guarantee when they are skipped locally.
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


# The WRITE schema is on the same rule. It was the last validator that was not:
# `DTCCreate.code` only checked 5 <= len <= 10, which is precisely how PEACE,
# PACED, P93AF, UA80E and UA80F entered the corpus through POST /api/v1/dtc/.
@pytest.mark.unit
@pytest.mark.parametrize("code", ["P0101", "p0300", "P26B7", "B00A0", " U0100 ", "P3AFF"])
def test_dtc_create_accepts_and_canonicalises_real_codes(code):
    from app.api.v1.schemas.dtc import DTCCreate

    created = DTCCreate(code=code, description_en="Real code", category="powertrain")
    assert created.code == code.strip().upper()


@pytest.mark.unit
@pytest.mark.parametrize(
    "code",
    ["PEACE", "PACED", "P93AF", "UA80E", "UA80F", "P9324", "U760E", "P8888", "X0300", "P030"],
)
def test_dtc_create_rejects_the_junk_that_seeded_the_corpus(code):
    from pydantic import ValidationError

    from app.api.v1.schemas.dtc import DTCCreate

    with pytest.raises(ValidationError):
        DTCCreate(code=code, description_en="Junk row", category="powertrain")


@pytest.mark.unit
def test_dtc_create_uses_the_shared_primitive_not_a_regex_of_its_own():
    """Guard against an eleventh copy of the DTC pattern.

    Ten divergent regexes were consolidated into `app.core.dtc_codes`; the
    schema module must import it, never restate it.
    """
    source = (BACKEND_DIR / "app" / "api" / "v1" / "schemas" / "dtc.py").read_text(encoding="utf-8")
    assert "from app.core.dtc_codes import" in source
    assert "[PBCU]" not in source, "a DTC regex was restated in the schema module"
    assert "re.compile" not in source


# ---------------------------------------------------------------------------
# The metrics endpoint normalizer shares the same rule
# ---------------------------------------------------------------------------
# This used to run in a subprocess: there were TWO metrics modules declaring
# colliding Prometheus Info metrics ("autocognitix_app_info"), so importing both
# into one interpreter raised `Duplicated timeseries in CollectorRegistry`, and
# the normalizer under test could not be imported next to `app.core.metrics`.
# The duplicate (`app/middleware/metrics.py`, never installed by `app.main` and
# therefore unimportable in any process that runs the app) has been deleted, so
# the surviving normalizer is just imported here.
@pytest.mark.unit
def test_endpoint_normalizer_uses_the_shared_dtc_rule():
    from unittest.mock import MagicMock

    from app.core.metrics import MetricsMiddleware

    normalize = MetricsMiddleware(app=MagicMock())._normalize_endpoint

    for code in ["P0300", "P26B7", "p0a94", "B00A0", "P0301"]:
        assert normalize(f"/api/v1/dtc/{code}") == "/api/v1/dtc/{dtc_code}", code

    # Junk is still NOT a code - but in the {code} position it collapses to a
    # single {invalid_dtc} series instead of one series per sprayed value. See
    # app/core/metrics_paths.py and tests/unit/test_metrics.py.
    for junk in ["PEACE", "UA80E", "P93AF", "P9324"]:
        assert normalize(f"/api/v1/dtc/{junk}") == "/api/v1/dtc/{invalid_dtc}", junk

    # ... while the literal sibling routes under /dtc keep their own labels.
    for literal in ["search", "categories", "bulk"]:
        assert normalize(f"/api/v1/dtc/{literal}") == f"/api/v1/dtc/{literal}", literal

    # A non-code segment OUTSIDE the {code} position is untouched: the SAE
    # tightening exists so a make named "PACED" keeps its own label.
    assert normalize("/api/v1/vehicles/PEACE/models") == "/api/v1/vehicles/PEACE/models"

    # other segment kinds must keep working
    assert normalize("/api/v1/vehicles/1HGBH41JXMN109186") == "/api/v1/vehicles/{vin}"
    assert normalize("/api/v1/items/12345") == "/api/v1/items/{id}"
    assert normalize("/health") == "/health"


@pytest.mark.unit
def test_the_dead_metrics_duplicate_stays_deleted():
    """Regression guard for the module removed alongside the test above.

    `app/middleware/metrics.py` was a second, never-installed copy of the
    metrics middleware. It declared the same Prometheus metric names as
    `app.core.metrics`, so it could not even be imported into a process that
    had imported the live module - it was unusable, not merely unused. If it
    comes back, the DTC path rule has two homes again and this file's premise
    (one rule, one definition) is false.
    """
    assert not (BACKEND_DIR / "app" / "middleware").exists(), (
        "app/middleware/ is back; it duplicated app/core/metrics.py and "
        "collided with it on the default Prometheus registry"
    )
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("app.middleware.metrics")


# ---------------------------------------------------------------------------
# The seed file is a write path, and it bypasses DTCCreate
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_shipped_seed_file_contains_no_unservable_code():
    """The seed file must not carry a code the rest of the API refuses to serve.

    `_seed_dtc_codes` inserts this file with raw SQL, so `DTCCreate`'s validator
    never sees it - this file IS the write path's contract. It historically
    shipped five rows (PEACE, PACED, P93AF, UA80E, UA80F) that `GET /dtc/{code}`
    answers 400 for.

    Migration 021 purges them, but a migration runs once. On a database that is
    still empty when it runs - a new environment, a staging rebuild, a restore -
    it purges nothing, stamps itself applied forever, and then seeding puts them
    straight back. Keeping the file itself clean is what makes the purge stick
    on environments that did not exist when it ran.
    """
    seed_file = BACKEND_DIR / "data" / "dtc_codes_seed.json"
    assert seed_file.exists(), f"seed file missing: {seed_file}"

    payload = json.loads(seed_file.read_text(encoding="utf-8"))
    rows = payload.get("codes", payload) if isinstance(payload, dict) else payload

    unservable = sorted(
        str(row.get("code", "")) for row in rows if not is_valid_dtc_code(str(row.get("code", "")))
    )
    assert not unservable, (
        f"{len(unservable)} seed row(s) fail the SAE J2012 rule and would be "
        f"re-inserted on any empty-database boot: {unservable[:25]}"
    )


@pytest.mark.unit
def test_seed_related_codes_never_point_at_an_unservable_code():
    """A suggestion the user cannot open is a dead link, seeded or not."""
    seed_file = BACKEND_DIR / "data" / "dtc_codes_seed.json"
    payload = json.loads(seed_file.read_text(encoding="utf-8"))
    rows = payload.get("codes", payload) if isinstance(payload, dict) else payload

    dangling = sorted(
        {
            related
            for row in rows
            for related in (row.get("related_codes") or [])
            if not is_valid_dtc_code(str(related))
        }
    )
    assert not dangling, f"seed related_codes reference unservable codes: {dangling[:25]}"
