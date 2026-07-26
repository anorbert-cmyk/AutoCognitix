"""
Unit tests for the DTC extraction used by the Neo4j complaint importer.

Covers the pure, database-free logic of `scripts/sync_neo4j_sprint9.py`:
  - the DTC regex (recall on real codes, rejection of the known false
    positives that polluted the graph and the curated code file),
  - the full-corpus scan (streaming, de-duplication, memory guard),
  - the relationship-budget ranking/capping helper,
  - checkpoint forward-compatibility.

It also asserts that `scripts/sync_nhtsa.py` (the second, independent
extraction pipeline) agrees with the canonical implementation, so the two
regexes cannot drift apart again.

No database, no network: the scripts are loaded by path with importlib.
"""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def _load_script(name: str) -> ModuleType:
    """Import a standalone script from scripts/ without installing it."""
    path = SCRIPTS_DIR / f"{name}.py"
    if not path.exists():  # pragma: no cover - repo layout guard
        pytest.skip(f"script not found: {path}")
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pytest.importorskip("neo4j", reason="neo4j driver required to import the sync script")
sync_neo4j = _load_script("sync_neo4j_sprint9")

pytest.importorskip("httpx", reason="httpx required to import sync_nhtsa")
sync_nhtsa = _load_script("sync_nhtsa")


# ---------------------------------------------------------------------------
# Regex recall: real DTC codes must be found
# ---------------------------------------------------------------------------
REAL_CODES = [
    # decimal codes the old strict pattern already found
    ("P0301", ["P0301"]),
    ("P0300", ["P0300"]),
    ("P0171", ["P0171"]),
    ("U0100", ["U0100"]),
    ("B0101", ["B0101"]),
    ("C1391", ["C1391"]),
    # hex codes the old strict pattern [PBCU][0-9]{4} silently dropped
    ("P26B7", ["P26B7"]),
    ("P090C", ["P090C"]),
    ("P0A94", ["P0A94"]),
    ("B00A0", ["B00A0"]),
    ("P17F1", ["P17F1"]),
    ("U3000", ["U3000"]),
    # manufacturer-specific second characters 1..3
    ("P1450", ["P1450"]),
    ("P324E", ["P324E"]),
]


@pytest.mark.unit
@pytest.mark.parametrize(("text", "expected"), REAL_CODES)
def test_real_codes_are_extracted(text, expected):
    assert sync_neo4j.extract_dtc_codes(text) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("p0171", ["P0171"]),
        ("Dealer found code p26b7 stored", ["P26B7"]),
        ("MIL ON, DTC: P0301", ["P0301"]),
        ("code P-0301 was pulled", ["P0301"]),
        ("scanner showed P 0300 misfire", ["P0300"]),
        ("codes p0301, P0302 and p0303", ["P0301", "P0302", "P0303"]),
    ],
)
def test_case_and_separator_handling(text, expected):
    assert sync_neo4j.extract_dtc_codes(text) == expected


# ---------------------------------------------------------------------------
# Regex precision: the documented false positives must be rejected
# ---------------------------------------------------------------------------
KNOWN_FALSE_POSITIVES = [
    # English words matched by the hex-permissive pattern in sync_nhtsa.py
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
def test_known_false_positives_are_rejected(token):
    assert sync_neo4j.extract_dtc_codes(token) == []
    assert sync_neo4j.extract_dtc_codes(f"the {token} appeared in the report") == []
    assert sync_neo4j.extract_dtc_codes(token.lower()) == []


@pytest.mark.unit
def test_false_positive_in_realistic_narrative():
    text = (
        "It has been reported on Nissan Service Campaign P9324, dated October 25, "
        "2019. The dealer also mentioned campaign PC861 for the headlamps."
    )
    assert sync_neo4j.extract_dtc_codes(text) == []


@pytest.mark.unit
def test_realistic_narrative_with_codes():
    text = (
        "THE CONTACT OWNS A 2015 FORD FOCUS. THE CHECK ENGINE LIGHT ILLUMINATED "
        "AND THE DEALER DIAGNOSED CODES P0301 AND p0304 (MISFIRE), PLUS P26B7."
    )
    assert sync_neo4j.extract_dtc_codes(text) == ["P0301", "P0304", "P26B7"]


# ---------------------------------------------------------------------------
# Boundaries
# ---------------------------------------------------------------------------
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
def test_codes_inside_longer_tokens_are_not_matched(text):
    assert sync_neo4j.extract_dtc_codes(text) == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("(P0301)", ["P0301"]),
        ("P0301.", ["P0301"]),
        ("P0301,P0302", ["P0301", "P0302"]),
        ("codes: P0301/P0420", ["P0301", "P0420"]),
        ("P0301-P0304", ["P0301", "P0304"]),
    ],
)
def test_punctuation_boundaries(text, expected):
    assert sync_neo4j.extract_dtc_codes(text) == expected


@pytest.mark.unit
@pytest.mark.parametrize("text", [None, "", "   ", "no codes here at all"])
def test_empty_and_codeless_text(text):
    assert sync_neo4j.extract_dtc_codes(text) == []


@pytest.mark.unit
def test_duplicate_codes_are_collapsed_and_sorted():
    text = "P0302 then P0301 then p0302 again"
    assert sync_neo4j.extract_dtc_codes(text) == ["P0301", "P0302"]


# ---------------------------------------------------------------------------
# is_valid_dtc_code + the curated code file
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize("code", ["P0301", "p0301", " U0100 ", "B00A0", "C3FFF"])
def test_is_valid_dtc_code_accepts_real_codes(code):
    assert sync_neo4j.is_valid_dtc_code(code) is True


@pytest.mark.unit
@pytest.mark.parametrize("code", [None, "", "PEACE", "UA80E", "P93AF", "P9324", "P030", "X0301"])
def test_is_valid_dtc_code_rejects_junk(code):
    assert sync_neo4j.is_valid_dtc_code(code) is False


@pytest.mark.unit
def test_curated_code_file_only_loses_the_known_junk():
    """The 6,814-code curated file holds exactly 5 non-DTC entries."""
    dtc_file = PROJECT_ROOT / "data" / "dtc_codes" / "all_codes_complete.json"
    if not dtc_file.exists():
        pytest.skip("curated DTC file not present in this checkout")
    with dtc_file.open() as f:
        codes = [str(c.get("code", "")) for c in json.load(f).get("codes", [])]
    rejected = {c for c in codes if not sync_neo4j.is_valid_dtc_code(c)}
    assert rejected == {"PEACE", "PACED", "P93AF", "UA80E", "UA80F"}
    # Everything else survives the structural filter.
    assert len(codes) - len(rejected) == len(sync_neo4j.load_curated_dtc_codes())


# ---------------------------------------------------------------------------
# The second pipeline must agree (no regex drift)
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize(
    "text",
    [
        "P0301 and P0302",
        "DTC: p26b7 stored",
        "trouble code P0A94",
        "PEACE of mind",
        "Nissan Service Campaign P9324",
        "transmission U760E shudder",
        "VIN 1FADP3F25FL",
        "error code B00A0 present",
    ],
)
def test_both_pipelines_extract_the_same_codes(text):
    canonical = set(sync_neo4j.extract_dtc_codes(text))
    assert sync_nhtsa.extract_dtc_codes(text) == canonical


# ---------------------------------------------------------------------------
# Corpus scan (full corpus, no safety bias)
# ---------------------------------------------------------------------------
def _complaint(odi, summary, deaths=0, date="20240101", make="FORD", model="FOCUS"):
    return {
        "odi_number": odi,
        "make": make,
        "model": model,
        "model_year": 2015,
        "component": "ENGINE",
        "summary": summary,
        "crash": False,
        "fire": False,
        "injuries": 0,
        "deaths": deaths,
        "date_received": date,
    }


@pytest.fixture()
def corpus_dir(tmp_path):
    (tmp_path / "a.json").write_text(
        json.dumps(
            {
                "complaints": [
                    # zero-safety-score complaint: the old safety-ranked 3%
                    # sample would never have reached it
                    _complaint(1, "check engine light, code P0301 and P0304"),
                    _complaint(2, "airbag light on", deaths=3),
                    _complaint(3, "Nissan Service Campaign P9324 refused"),
                ]
            }
        )
    )
    (tmp_path / "b.json").write_text(
        json.dumps(
            {
                "complaints": [
                    _complaint(4, "dealer pulled p26b7", date="20200101"),
                    # duplicate odi across files must not duplicate the node
                    _complaint(1, "code P0301 again", date="20240101"),
                ]
            }
        )
    )
    return tmp_path


@pytest.mark.unit
def test_scan_reads_every_complaint_regardless_of_safety_score(corpus_dir):
    scan = sync_neo4j.scan_corpus_for_dtc_mentions(corpus_dir, ["a.json", "b.json"], verbose=False)
    assert scan.stats["complaints_scanned"] == 5
    assert scan.stats["complaints_with_dtc"] == 2  # odi 1 and 4 (odi 1 de-duped)
    assert scan.stats["unique_codes"] == 3  # P0301, P0304, P26B7
    assert scan.stats["truncated"] is False
    assert sorted(scan.complaint_nodes) == ["1", "4"]
    assert {(p["odi_id"], p["code"]) for p in scan.pairs} == {
        ("1", "P0301"),
        ("1", "P0304"),
        ("4", "P26B7"),
    }


@pytest.mark.unit
def test_scan_ignores_campaign_ids(corpus_dir):
    scan = sync_neo4j.scan_corpus_for_dtc_mentions(corpus_dir, ["a.json"], verbose=False)
    assert "3" not in scan.complaint_nodes


@pytest.mark.unit
def test_scan_tolerates_missing_files(corpus_dir):
    scan = sync_neo4j.scan_corpus_for_dtc_mentions(
        corpus_dir, ["a.json", "missing.json"], verbose=False
    )
    assert scan.stats["complaints_scanned"] == 3


@pytest.mark.unit
def test_scan_memory_guard_truncates(corpus_dir):
    scan = sync_neo4j.scan_corpus_for_dtc_mentions(
        corpus_dir, ["a.json", "b.json"], max_pairs=1, verbose=False
    )
    assert scan.stats["truncated"] is True
    # The guard trips after the complaint that crossed the bound, so the whole
    # corpus is not held in memory: scanning stopped at the first record.
    assert scan.stats["complaints_scanned"] == 1
    assert len(scan.pairs) == 2


@pytest.mark.unit
def test_scan_accepts_bare_array_files(tmp_path):
    (tmp_path / "flat.json").write_text(json.dumps([_complaint(9, "code U0100")]))
    scan = sync_neo4j.scan_corpus_for_dtc_mentions(tmp_path, ["flat.json"], verbose=False)
    assert scan.stats["mentions"] == 1


@pytest.mark.unit
def test_normalize_complaint_maps_node_properties():
    node = sync_neo4j.normalize_complaint(
        {
            "odi_number": 42,
            "make": "ford",
            "model": "focus",
            "model_year": "2015",
            "summary": "x" * 6000,
            "injuries": None,
            "deaths": "2",
        }
    )
    assert node["odi_id"] == "42"
    assert node["make"] == "FORD"
    assert node["model"] == "FOCUS"
    assert node["year"] == 2015
    assert len(node["description"]) == 5000
    assert node["injuries"] == 0
    assert node["deaths"] == 2


# ---------------------------------------------------------------------------
# Relationship budget: ranking + capping
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_prioritize_prefers_curated_codes_then_newest():
    pairs = [
        {"odi_id": "1", "code": "P9999X", "date_received": "20250101"},
        {"odi_id": "2", "code": "P0301", "date_received": "20100101"},
        {"odi_id": "3", "code": "P0420", "date_received": "20240101"},
    ]
    ranked = sync_neo4j.prioritize_dtc_pairs(pairs, {"P0301", "P0420"}, limit=3)
    assert [p["odi_id"] for p in ranked] == ["3", "2", "1"]


# ---------------------------------------------------------------------------
# _date_rank: the format on disk is ISO, not digits
#
# REGRESSION. `_date_rank` was `value if value.isdigit() else ""`, but
# import_flat_complaints.parse_date() writes "YYYY-MM-DD" into every record of
# the corpus (and therefore into Complaint.date_received in Neo4j). So EVERY
# real complaint returned "" -> one undifferentiated "unknown" bucket -> the
# "newest complaints first" half of the ranking never ran, while the capping log
# message kept announcing it. The whole test suite missed it because every
# fixture below used the compact "YYYYMMDD" shape, which exists only BEFORE
# parse_date() runs. These tests use the shape that is actually on disk.
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2020-03-15", "20200315"),  # THE format on disk (parse_date output)
        ("2024-07-31", "20240731"),
        ("2020-03-15T00:00:00", "20200315"),  # ISO datetime / neo4j Date repr
        ("20240731", "20240731"),  # raw NHTSA FLAT_CMPL field
    ],
)
def test_date_rank_parses_the_formats_that_actually_occur(raw, expected):
    assert sync_neo4j._date_rank(raw) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw", ["", None, "not-a-date", "2020-13-45", "0000-00-00", "1899-12-31", "2020-03", 12345]
)
def test_date_rank_rejects_missing_and_malformed_dates(raw):
    """Unknown is its own bucket - never a value that can out-rank a real date."""
    assert sync_neo4j._date_rank(raw) == ""


@pytest.mark.unit
def test_unknown_dates_sort_strictly_after_every_real_date():
    """`""` must be a predictable LAST, not "indistinguishable from everything".

    A real date always inverts to a leading digit <= 8 (year >= 1900), so the
    all-nines unknown key can never interleave with dated records.
    """
    assert sync_neo4j._invert_date("") == "9" * sync_neo4j._DATE_KEY_WIDTH
    for real in ("19000101", "20240731", "21001231"):
        assert sync_neo4j._invert_date(real) < sync_neo4j._invert_date("")


@pytest.mark.unit
def test_prioritize_ranks_iso_dates_newest_first():
    """THE regression: with ISO dates the cap must keep the NEWEST mentions.

    Before the fix every date parsed to "" and the tie broke on odi_id, so the
    cap kept the LOWEST odi_ids - i.e. the OLDEST complaints, the exact opposite
    of what the docstring and the capping log message promise.
    """
    pairs = [
        {"odi_id": "900", "code": "P0301", "date_received": "2024-07-31"},
        {"odi_id": "100", "code": "P0301", "date_received": "2005-01-01"},
        {"odi_id": "500", "code": "P0301", "date_received": "2015-11-02"},
    ]
    ranked = sync_neo4j.prioritize_dtc_pairs(pairs, {"P0301"}, limit=3)
    assert [p["odi_id"] for p in ranked] == ["900", "500", "100"]

    # And the cap keeps the newest, not the numerically smallest odi_id.
    assert [p["odi_id"] for p in sync_neo4j.prioritize_dtc_pairs(pairs, {"P0301"}, 1)] == ["900"]


@pytest.mark.unit
def test_prioritize_puts_undated_mentions_last_even_with_a_low_odi_id():
    pairs = [
        {"odi_id": "001", "code": "P0301", "date_received": ""},
        {"odi_id": "900", "code": "P0301", "date_received": "2024-07-31"},
    ]
    ranked = sync_neo4j.prioritize_dtc_pairs(pairs, {"P0301"}, limit=2)
    assert [p["odi_id"] for p in ranked] == ["900", "001"]


@pytest.mark.unit
def test_prioritize_warns_when_no_date_can_be_parsed(capsys):
    """The failure mode must announce itself instead of degrading in silence."""
    pairs = [{"odi_id": "1", "code": "P0301", "date_received": "15/03/2020"}]
    sync_neo4j.prioritize_dtc_pairs(pairs, {"P0301"}, limit=1)
    out = capsys.readouterr().out
    assert "INACTIVE" in out and "date_received" in out


@pytest.mark.unit
def test_prioritize_does_not_warn_when_dates_parse(capsys):
    pairs = [{"odi_id": "1", "code": "P0301", "date_received": "2020-03-15"}]
    sync_neo4j.prioritize_dtc_pairs(pairs, {"P0301"}, limit=1)
    assert "INACTIVE" not in capsys.readouterr().out


@pytest.mark.unit
def test_prioritize_caps_at_limit():
    pairs = [{"odi_id": str(i), "code": "P0301", "date_received": "20240101"} for i in range(10)]
    assert len(sync_neo4j.prioritize_dtc_pairs(pairs, {"P0301"}, limit=4)) == 4
    assert sync_neo4j.prioritize_dtc_pairs(pairs, {"P0301"}, limit=0) == []
    assert sync_neo4j.prioritize_dtc_pairs(pairs, {"P0301"}, limit=-5) == []


@pytest.mark.unit
def test_prioritize_is_deterministic_and_handles_missing_dates():
    pairs = [
        {"odi_id": "b", "code": "P0301"},
        {"odi_id": "a", "code": "P0301", "date_received": "20240101"},
    ]
    ranked = sync_neo4j.prioritize_dtc_pairs(pairs, {"P0301"}, limit=2)
    assert [p["odi_id"] for p in ranked] == ["a", "b"]
    assert sync_neo4j.prioritize_dtc_pairs(pairs, {"P0301"}, limit=2) == ranked


@pytest.mark.unit
def test_dtc_rel_limit_fits_the_aura_free_cap():
    """The configured per-run cap plus the reserve must stay under 400K."""
    assert (
        sync_neo4j.DEFAULT_DTC_REL_LIMIT + sync_neo4j.REL_RESERVE_FOR_LATER_STEPS
        <= sync_neo4j.AURA_FREE_REL_CAP
    )


@pytest.mark.unit
def test_counters_helper_reads_write_counters():
    class _Counters:
        nodes_created = 3
        relationships_created = 7

    class _Summary:
        counters = _Counters()

    assert sync_neo4j.Neo4jSprint9Loader._counters(_Summary()) == (3, 7)
    assert sync_neo4j.Neo4jSprint9Loader._counters(None) == (0, 0)


@pytest.mark.unit
def test_dtc_category_mapping():
    assert sync_neo4j.dtc_category("P0301") == "powertrain"
    assert sync_neo4j.dtc_category("B0101") == "body"
    assert sync_neo4j.dtc_category("C1391") == "chassis"
    assert sync_neo4j.dtc_category("U0100") == "network"
    assert sync_neo4j.dtc_category("") == "unknown"


# ---------------------------------------------------------------------------
# Checkpoint forward-compatibility (resume must survive a schema change)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_checkpoint_merges_new_keys_into_an_old_file(tmp_path):
    cp_file = tmp_path / "cp.json"
    cp_file.write_text(json.dumps({"dtc_loaded": True}))  # pre-upgrade checkpoint
    manager = sync_neo4j.CheckpointManager(cp_file)
    assert manager.state["dtc_loaded"] is True
    assert manager.state["dtc_complaint_rels"] is False
    assert manager.state["dtc_complaint_rels_created"] == 0


@pytest.mark.unit
def test_checkpoint_clear_reenables_a_step(tmp_path):
    cp_file = tmp_path / "cp.json"
    manager = sync_neo4j.CheckpointManager(cp_file)
    manager.mark_complete("dtc_complaint_rels")
    assert manager.state["dtc_complaint_rels"] is True
    manager.clear("dtc_complaint_rels")
    assert manager.state["dtc_complaint_rels"] is False
    assert json.loads(cp_file.read_text())["dtc_complaint_rels"] is False


# ---------------------------------------------------------------------------
# Checkpoint: a run that did NOTHING must not record success
#
# REGRESSION. When the flat-file corpus is absent the step falls back to
# scanning the in-graph sample; on an empty/unloaded graph that yields zero
# pairs, and the code then called mark_complete("dtc_complaint_rels"). The
# checkpoint file persists, so every LATER run - including one on a properly
# provisioned machine with the full 1.66M-record corpus - printed
# "[SKIP] ... already created" and never ran the extraction. A degraded run's
# empty result masqueraded as a finished step, and the skip message made it look
# deliberate. This is the same class of bug as the zero-vector embedding.
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_is_complete_reads_legacy_boolean_checkpoints(tmp_path):
    cp_file = tmp_path / "cp.json"
    cp_file.write_text(json.dumps({"dtc_complaint_rels": True}))
    manager = sync_neo4j.CheckpointManager(cp_file)
    assert manager.is_complete("dtc_complaint_rels") is True
    assert manager.is_complete("vehicles_loaded") is False


@pytest.mark.unit
def test_degraded_record_is_persisted_but_is_not_complete(tmp_path):
    """The record must be readable by an operator AND not count as done."""
    cp_file = tmp_path / "cp.json"
    manager = sync_neo4j.CheckpointManager(cp_file)
    manager.mark_degraded("dtc_complaint_rels", reason="corpus_missing", created=0)

    assert manager.is_complete("dtc_complaint_rels") is False
    stored = json.loads(cp_file.read_text())["dtc_complaint_rels"]
    assert stored["complete"] is False
    assert stored["reason"] == "corpus_missing"
    assert stored["created"] == 0
    assert stored["at"]

    # A fresh manager (i.e. the NEXT run) must reach the same conclusion.
    assert sync_neo4j.CheckpointManager(cp_file).is_complete("dtc_complaint_rels") is False


@pytest.mark.unit
def test_degraded_record_is_truthy_so_is_complete_is_mandatory(tmp_path):
    """Why every call site had to move off `if state[key]:`.

    A degraded record is a non-empty dict. Raw truthiness reads it as "done" -
    exactly the bug, re-introduced. This test fails the moment someone reverts a
    call site to a plain truthiness check.
    """
    cp_file = tmp_path / "cp.json"
    manager = sync_neo4j.CheckpointManager(cp_file)
    manager.mark_degraded("dtc_complaint_rels", reason="corpus_missing")

    assert bool(manager.state["dtc_complaint_rels"]) is True  # the trap
    assert manager.is_complete("dtc_complaint_rels") is False  # the guard

    source = (PROJECT_ROOT / "scripts" / "sync_neo4j_sprint9.py").read_text(encoding="utf-8")
    for key in (
        "dtc_loaded",
        "vehicles_loaded",
        "engines_loaded",
        "complaints_loaded",
        "dtc_complaint_rels",
        "vehicle_complaint_rels",
        "vehicle_engine_rels",
    ):
        assert f'checkpoint.state["{key}"]:' not in source, (
            f"{key} is read through raw truthiness again - use is_complete()"
        )
        assert f'checkpoint.state.get("{key}"):' not in source


@pytest.mark.unit
def test_clear_resets_a_degraded_record_too(tmp_path):
    cp_file = tmp_path / "cp.json"
    manager = sync_neo4j.CheckpointManager(cp_file)
    manager.mark_degraded("dtc_complaint_rels", reason="corpus_missing")
    manager.clear("dtc_complaint_rels")
    assert manager.state["dtc_complaint_rels"] is False
    assert manager.is_complete("dtc_complaint_rels") is False


def _degraded_loader(cp_file, pairs):
    """A loader whose only live parts are the checkpoint and the fallback scan."""
    loader = sync_neo4j.Neo4jSprint9Loader.__new__(sync_neo4j.Neo4jSprint9Loader)
    loader.checkpoint = sync_neo4j.CheckpointManager(cp_file)
    loader.stats = dict.fromkeys(
        [
            "dtc",
            "vehicles",
            "engines",
            "complaints",
            "dtc_complaint_rels",
            "dtc_complaint_nodes_created",
            "dtc_nodes_created",
            "vehicle_complaint_rels",
            "vehicle_engine_rels",
        ],
        0,
    )

    async def _fake_graph_scan():
        return sync_neo4j.DtcScanResult(
            pairs=list(pairs),
            complaint_nodes={},
            stats={
                "complaints_scanned": 0,
                "complaints_with_dtc": 0,
                "mentions": len(pairs),
                "unique_codes": 0,
                "hit_rate": 0.0,
                "truncated": False,
                "source": "neo4j-nodes",
            },
        )

    loader._scan_graph_for_dtc_mentions = _fake_graph_scan
    return loader


@pytest.mark.unit
@pytest.mark.asyncio
async def test_missing_corpus_zero_rels_does_not_close_the_step(tmp_path, monkeypatch):
    """THE regression, end to end: no corpus + empty graph => step stays open."""
    monkeypatch.setattr(sync_neo4j, "DATA_DIR", tmp_path / "absent")
    cp_file = tmp_path / "cp.json"

    loader = _degraded_loader(cp_file, pairs=[])
    await loader.create_dtc_complaint_relationships()

    assert loader.checkpoint.is_complete("dtc_complaint_rels") is False
    assert json.loads(cp_file.read_text())["dtc_complaint_rels"]["reason"] == "corpus_missing"

    # The next run must actually RUN the step, not print "[SKIP] already created".
    assert sync_neo4j.CheckpointManager(cp_file).is_complete("dtc_complaint_rels") is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_full_corpus_with_genuinely_zero_mentions_does_close_the_step(tmp_path, monkeypatch):
    """A real scan that legitimately finds nothing IS a completed step.

    This is the distinction the fix exists to make: an empty ANSWER from a scan
    that really ran over the corpus is a result; an empty answer because the
    corpus was missing is a failure. They must not be recorded the same way.
    """
    complaints_dir = tmp_path / "nhtsa" / "complaints_flat"
    complaints_dir.mkdir(parents=True)
    # Real corpus file, real scan, no DTC code anywhere in it.
    (complaints_dir / sync_neo4j.COMPLAINT_FILES[0]).write_text(
        json.dumps({"complaints": [_complaint(1, "the brakes squeal when cold")]})
    )
    monkeypatch.setattr(sync_neo4j, "DATA_DIR", tmp_path)
    cp_file = tmp_path / "cp.json"

    loader = _degraded_loader(cp_file, pairs=[])
    await loader.create_dtc_complaint_relationships()

    assert loader.checkpoint.is_complete("dtc_complaint_rels") is True


@pytest.mark.unit
def test_missing_complaint_data_does_not_close_the_load_step(tmp_path, monkeypatch, capsys):
    """Same bug, sibling call site: load_complaints() marked itself complete."""
    monkeypatch.setattr(sync_neo4j, "DATA_DIR", tmp_path / "absent")
    cp_file = tmp_path / "cp.json"
    loader = _degraded_loader(cp_file, pairs=[])

    asyncio.run(loader.load_complaints())

    assert loader.checkpoint.is_complete("complaints_loaded") is False
    assert json.loads(cp_file.read_text())["complaints_loaded"]["reason"] == "no_complaint_data"
    assert "NOT marked complete" in capsys.readouterr().out


@pytest.mark.unit
def test_missing_engine_specs_does_not_close_the_engine_rel_step(tmp_path, monkeypatch):
    """Same bug, second sibling call site."""
    monkeypatch.setattr(sync_neo4j, "DATA_DIR", tmp_path / "absent")
    cp_file = tmp_path / "cp.json"
    loader = _degraded_loader(cp_file, pairs=[])

    asyncio.run(loader.create_vehicle_engine_relationships())

    assert loader.checkpoint.is_complete("vehicle_engine_rels") is False
    assert (
        json.loads(cp_file.read_text())["vehicle_engine_rels"]["reason"] == "engine_specs_missing"
    )
