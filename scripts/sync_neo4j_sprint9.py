#!/usr/bin/env python3
"""
Sprint 9 Neo4j Data Sync.
Loads DTC codes, vehicles, engines, complaints into Neo4j Aura.
Uses async neo4j driver with batch UNWIND + MERGE patterns.

Usage:
    NEO4J_URI=neo4j+s://xxx NEO4J_PASSWORD=xxx python scripts/sync_neo4j_sprint9.py --all
    NEO4J_URI=neo4j+s://xxx NEO4J_PASSWORD=xxx python scripts/sync_neo4j_sprint9.py --dtc
    NEO4J_URI=neo4j+s://xxx NEO4J_PASSWORD=xxx python scripts/sync_neo4j_sprint9.py --vehicles
    NEO4J_URI=neo4j+s://xxx NEO4J_PASSWORD=xxx python scripts/sync_neo4j_sprint9.py --engines
    NEO4J_URI=neo4j+s://xxx NEO4J_PASSWORD=xxx python scripts/sync_neo4j_sprint9.py --complaints
    NEO4J_URI=neo4j+s://xxx NEO4J_PASSWORD=xxx python scripts/sync_neo4j_sprint9.py --reset

    # DB-free dry run: scan the whole complaint corpus and report what the
    # DTC extraction WOULD create (no credentials needed, nothing is written):
    python scripts/sync_neo4j_sprint9.py --scan-only

    # Re-run only the (improved) DTC extraction step on an existing graph:
    NEO4J_URI=... NEO4J_PASSWORD=... python scripts/sync_neo4j_sprint9.py \\
        --complaints --redo-dtc-rels --dtc-rel-limit 60000
"""

import argparse
import asyncio
import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Sequence, Set, Tuple

from neo4j import AsyncGraphDatabase

try:  # pragma: no cover - trivial import shim
    from tqdm import tqdm
except ImportError:  # tqdm is optional: keeps the pure helpers importable in CI

    class tqdm:  # type: ignore[no-redef]
        """Minimal no-op stand-in so the module imports without tqdm."""

        def __init__(self, total: int = 0, desc: str = "", unit: str = "") -> None:
            self.total = total
            self.desc = desc

        def __enter__(self) -> "tqdm":
            return self

        def __exit__(self, *exc: Any) -> None:
            print(f"  {self.desc}: done ({self.total:,})")

        def update(self, n: int = 1) -> None:
            pass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

BATCH_SIZE = 500
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds, exponentially increased

# How many complaints become Complaint NODES (safety-ranked). This is a node
# budget decision and is deliberately NOT the corpus the DTC extraction runs on
# - see scan_corpus_for_dtc_mentions() for why the two are decoupled.
COMPLAINT_LIMIT = 50_000

# --- Neo4j Aura Free capacity guards -------------------------------------
# Aura Free allows 400K relationships in total and the project has already run
# into that ceiling once (docs/COWORK_BRIEF.md). Every relationship-creating
# step below has to fit into the REMAINING headroom, not into a fresh 400K.
AURA_FREE_REL_CAP = 400_000
# Reserved for the steps that run AFTER the DTC extraction in run():
# HAS_COMPLAINT (<= ~50K, one per complaint that has a matching Vehicle) plus
# USES_ENGINE (~30K unique (vehicle, engine) pairs) plus margin.
REL_RESERVE_FOR_LATER_STEPS = 100_000
# Hard ceiling for MENTIONS_DTC edges created in one run. The measured corpus
# yield is ~28K (see scan_corpus_for_dtc_mentions), so this is ~2x headroom
# and still leaves the graph far away from the cap.
DEFAULT_DTC_REL_LIMIT = 60_000
# Defensive bound on scan memory: stop collecting mentions past this many pairs.
DTC_SCAN_MAX_PAIRS = 250_000
# Warn when the graph passes this fraction of the Aura Free relationship cap.
REL_WARN_FRACTION = 0.9

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
CHECKPOINT_FILE = SCRIPT_DIR / "checkpoints" / "neo4j_sprint9.json"

# DTC rules (SAE J2012) - single source of truth, IMPORTED not copied.
# backend/app/core/dtc_codes.py carries the full rationale and the measured
# false-positive numbers. `app.core` no longer builds the FastAPI Settings
# object on import, so this works with no .env and no SECRET_KEY.
sys.path.insert(0, str(PROJECT_DIR / "backend"))

from app.core.dtc_codes import (  # noqa: E402
    dtc_category,
    extract_dtc_codes,
    is_valid_dtc_code,
)

# Marks Complaint nodes that exist ONLY because a DTC code was extracted from
# them (they are outside the safety-ranked top-50K node set).
COMPLAINT_SOURCE_EXTRACTION = "dtc_extraction"
# Marks DTC nodes that were NOT in the curated 6.8K code set but appeared in a
# complaint narrative. Curated nodes have no `source` property.
DTC_SOURCE_EXTRACTION = "complaint_extraction"

# Complaint flat-file names in chronological order
COMPLAINT_FILES = [
    "2020-2024.json",
    "2025-2026.json",
    "2015-2019.json",
    "2010-2014.json",
    "2005-2009.json",
    "2000-2004.json",
]


# ---------------------------------------------------------------------------
# DTC extraction (pure functions - unit tested in
# backend/tests/unit/test_dtc_extraction.py, no database required)
# ---------------------------------------------------------------------------
#
# These used to be re-implemented here, and this script was documented as "the
# canonical reference implementation". It is not any more: the SAE J2012 rules
# (including the measured false-positive numbers and why the second character
# must be 0-3) live in backend/app/core/dtc_codes.py, and are imported above.
# One definition, no hand-syncing.


def load_curated_dtc_codes() -> Set[str]:
    """
    Load the curated DTC code set from disk (no database needed).

    Structurally invalid entries are dropped so the ranking below cannot
    "prefer" the junk codes that the old loose regex wrote into the file.
    """
    dtc_file = DATA_DIR / "dtc_codes" / "all_codes_complete.json"
    if not dtc_file.exists():
        return set()
    with dtc_file.open() as f:
        data = json.load(f)
    return {
        str(c.get("code", "")).upper()
        for c in data.get("codes", [])
        if is_valid_dtc_code(c.get("code"))
    }


# Width of the normalized sort key produced by _date_rank ("YYYYMMDD").
_DATE_KEY_WIDTH = 8


def _date_rank(date_received: Any) -> str:
    """
    Normalize a complaint date into a sortable ``YYYYMMDD`` key.

    THE TWO SHAPES THAT ACTUALLY REACH THIS FUNCTION:
      - ``"YYYY-MM-DD"`` - what ``import_flat_complaints.parse_date()`` writes
        into every record of the flat-file corpus, and therefore also what
        ``Complaint.date_received`` holds in Neo4j (the in-graph fallback scan
        reads it straight back out). This is the format on disk TODAY.
      - ``"YYYYMMDD"``   - the raw NHTSA ``FLAT_CMPL`` field, i.e. the input to
        ``parse_date()``. Accepted so an unparsed record still ranks correctly.

    The previous implementation was ``value if value.isdigit() else ""``, which
    returns ``""`` for EVERY hyphenated date - that is, for every real complaint.
    Every record then collapsed into the single "unknown" bucket, silently
    disabling the "newest first" half of :func:`prioritize_dtc_pairs` while the
    docstring and the capping log message kept claiming it was active.

    Args:
        date_received: Raw date value of a complaint (any type; ``None`` ok).

    Returns:
        str: ``"YYYYMMDD"`` for a parseable in-range date, or ``""`` for a
        missing/malformed one. ``""`` is NOT "sorts like everything else": it is
        its own bucket that :func:`_invert_date` places strictly after every
        real date, so unknown-dated mentions are the first to be dropped by the
        cap and never displace a dated one.
    """
    value = str(date_received or "").strip()
    if not value:
        return ""

    if len(value) >= 10 and value[4] == "-" and value[7] == "-":
        # "YYYY-MM-DD", optionally with a time suffix ("...T00:00:00").
        digits = value[:4] + value[5:7] + value[8:10]
    elif len(value) == _DATE_KEY_WIDTH:
        digits = value
    else:
        return ""

    if not digits.isdigit():
        return ""

    year, month, day = int(digits[:4]), int(digits[4:6]), int(digits[6:8])
    # Same range discipline as import_flat_complaints.parse_date(): a structurally
    # digit-shaped but nonsensical date must not out-rank a real one.
    if not (1900 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31):
        return ""
    return digits


def prioritize_dtc_pairs(
    pairs: Sequence[Dict[str, Any]],
    curated_codes: Set[str],
    limit: int,
) -> List[Dict[str, Any]]:
    """
    Rank (complaint, code) mentions and cap them at `limit`.

    Ordering: curated codes first (they carry descriptions/severity and are the
    ones the UI can actually render), then most recent complaints, then odi_id
    for a deterministic result. Capping is what keeps the Aura Free
    relationship budget predictable - see AURA_FREE_REL_CAP.
    """
    if limit <= 0:
        return []

    # Self-check for the exact bug this function shipped with: if NOTHING has a
    # parseable date, the recency tier is inert and the cap silently degrades to
    # odi_id order. Say so instead of printing "newest complaints first" over a
    # ranking that is nothing of the sort.
    if pairs and not any(_date_rank(p.get("date_received")) for p in pairs):
        print(
            f"  [WARN] None of the {len(pairs):,} mentions carry a parseable "
            "date_received (expected YYYY-MM-DD or YYYYMMDD). The 'newest first' "
            "ranking is INACTIVE - the cap falls back to odi_id order."
        )

    ranked = sorted(
        pairs,
        key=lambda p: (
            0 if str(p.get("code", "")).upper() in curated_codes else 1,
            _invert_date(_date_rank(p.get("date_received"))),
            str(p.get("odi_id", "")),
            str(p.get("code", "")),
        ),
    )
    return list(ranked[:limit])


def _invert_date(value: str) -> str:
    """
    Map a ``YYYYMMDD`` key from :func:`_date_rank` to a newest-first sort key.

    Unknown dates return all-nines, which is strictly greater than any inverted
    real date (a year >= 1900 inverts to a leading digit <= 8), so they always
    sort last - deterministically, never interleaved with dated records.
    """
    if not value:
        return "9" * _DATE_KEY_WIDTH  # unknown / unparseable dates sort last
    return "".join(str(9 - int(ch)) for ch in value)


def normalize_complaint(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Map a raw NHTSA flat-file complaint to Neo4j Complaint node properties."""
    return {
        "odi_id": str(raw.get("odi_number", "")),
        "make": (raw.get("make") or "").upper(),
        "model": (raw.get("model") or "").upper(),
        "year": int(raw.get("model_year") or 0),
        "component": raw.get("component", "") or "",
        "description": (raw.get("summary") or "")[:5000],
        "crash": bool(raw.get("crash")),
        "fire": bool(raw.get("fire")),
        "injuries": int(raw.get("injuries") or 0),
        "deaths": int(raw.get("deaths") or 0),
        "date_received": raw.get("date_received", "") or "",
    }


def iter_corpus_complaints(
    complaints_dir: Path,
    filenames: Sequence[str] = tuple(COMPLAINT_FILES),
    verbose: bool = True,
) -> Iterator[Dict[str, Any]]:
    """
    Stream every complaint of the raw flat-file corpus, ONE FILE AT A TIME.

    Peak memory is bounded by the largest single file instead of the whole
    corpus (the old fallback in _collect_complaints_sorted concatenated all of
    them before sorting).
    """
    for fname in filenames:
        fpath = complaints_dir / fname
        if not fpath.exists():
            if verbose:
                print(f"  [WARN] Missing corpus file: {fname}")
            continue
        if verbose:
            print(f"  Scanning {fname} ...")
        with fpath.open() as f:
            data = json.load(f)
        rows = data.get("complaints", []) if isinstance(data, dict) else data
        yield from rows
        del rows, data
        gc.collect()


class DtcScanResult(NamedTuple):
    """Output of the corpus-wide DTC extraction pass."""

    pairs: List[Dict[str, Any]]  # {odi_id, code, date_received}
    complaint_nodes: Dict[str, Dict[str, Any]]  # odi_id -> node properties
    stats: Dict[str, Any]


def scan_corpus_for_dtc_mentions(
    complaints_dir: Path,
    filenames: Sequence[str] = tuple(COMPLAINT_FILES),
    max_pairs: int = DTC_SCAN_MAX_PAIRS,
    verbose: bool = True,
) -> DtcScanResult:
    """
    Run the DTC extraction over the FULL complaint corpus (~1.66M records).

    WHY THE FULL CORPUS: extraction is a cheap, read-only text pass and is
    completely independent of which complaints become NODES. The node set is
    capped at COMPLAINT_LIMIT and ranked by _safety_score (deaths > injuries >
    fire > crash) - i.e. deliberately airbag / seat-belt / structure complaints,
    the one category that essentially never quotes a powertrain DTC. Extracting
    only from that 3% sample is what produced the ~107 MENTIONS_DTC edges the
    live graph has. Scanning everything and creating nodes only for the
    complaints that actually mention a code keeps the node budget intact while
    multiplying the edge yield.

    HONEST EXPECTATION - do not oversell this: measured on 26,237 real NHTSA
    narratives only 1.28% of complaints mention any DTC at all, at ~0.0167
    mentions per complaint. Extrapolated to the 1,656,899 parsed complaints that
    is roughly 21K complaints / 28K MENTIONS_DTC edges (+/- a few thousand,
    depending on how representative the sample is). That is ~250x today's 107
    edges, but it is still only ~1.3% of the corpus: consumers describe
    symptoms, not codes. This track stays SUPPLEMENTARY to the PostgreSQL
    component-frequency ranking. Run `--scan-only` to get the exact number for
    the local corpus before touching the database.
    """
    pairs: List[Dict[str, Any]] = []
    complaint_nodes: Dict[str, Dict[str, Any]] = {}
    scanned = 0
    matched = 0
    truncated = False
    codes_seen: Dict[str, int] = {}

    for raw in iter_corpus_complaints(complaints_dir, filenames, verbose=verbose):
        scanned += 1
        codes = extract_dtc_codes(raw.get("summary") or raw.get("description"))
        if not codes:
            continue
        node = normalize_complaint(raw)
        odi_id = node["odi_id"]
        if not odi_id:
            continue
        if odi_id not in complaint_nodes:
            complaint_nodes[odi_id] = node
            matched += 1
        for code in codes:
            pairs.append(
                {
                    "odi_id": odi_id,
                    "code": code,
                    "date_received": node["date_received"],
                }
            )
            codes_seen[code] = codes_seen.get(code, 0) + 1
        if len(pairs) >= max_pairs:
            truncated = True
            print(f"  [WARN] Scan stopped at {max_pairs:,} mentions (memory guard).")
            break

    stats = {
        "complaints_scanned": scanned,
        "complaints_with_dtc": matched,
        "mentions": len(pairs),
        "unique_codes": len(codes_seen),
        "hit_rate": (matched / scanned) if scanned else 0.0,
        "truncated": truncated,
    }
    return DtcScanResult(pairs=pairs, complaint_nodes=complaint_nodes, stats=stats)


def print_scan_report(scan: DtcScanResult) -> None:
    """Human-readable summary of an extraction pass (also used by --scan-only)."""
    stats = scan.stats
    curated = load_curated_dtc_codes()
    codes = {p["code"] for p in scan.pairs}
    non_curated = sorted(codes - curated)
    print()
    print("  --- DTC extraction report -----------------------------------")
    print(f"  source                : {stats.get('source', 'flat-files')}")
    print(f"  complaints scanned    : {stats['complaints_scanned']:,}")
    print(
        f"  complaints with a DTC : {stats['complaints_with_dtc']:,} "
        f"({100 * stats['hit_rate']:.2f}%)"
    )
    print(f"  DTC mentions (edges)  : {stats['mentions']:,}")
    print(f"  unique codes          : {stats['unique_codes']:,}")
    print(f"  codes outside curated : {len(non_curated):,}")
    if non_curated:
        print(f"    e.g. {', '.join(non_curated[:12])}")
    if stats.get("truncated"):
        print("  [WARN] scan hit DTC_SCAN_MAX_PAIRS - results are truncated")
    print("  ------------------------------------------------------------")
    print()


# ---------------------------------------------------------------------------
# Checkpoint Manager
# ---------------------------------------------------------------------------
class CheckpointManager:
    """Persist progress across runs so we can resume after failures."""

    def __init__(self, checkpoint_file: Path) -> None:
        self.checkpoint_file = checkpoint_file
        self.state: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        state = self._defaults()
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file) as f:
                stored = json.load(f)
            # Merge over the defaults so a checkpoint written by an older
            # version of this script never KeyErrors on a newly added key.
            if isinstance(stored, dict):
                state.update(stored)
        return state

    @staticmethod
    def _defaults() -> Dict[str, Any]:
        return {
            "dtc_loaded": False,
            "vehicles_loaded": False,
            "engines_loaded": False,
            "complaints_loaded": False,
            "complaints_files_done": [],
            "complaints_current_file": None,
            "complaints_current_index": 0,
            "dtc_complaint_rels": False,
            "dtc_complaint_rels_created": 0,
            "vehicle_complaint_rels": False,
            "vehicle_engine_rels": False,
            "last_updated": None,
        }

    def is_complete(self, key: str) -> bool:
        """
        True only when `key` records a run that ACTUALLY did its work.

        ALWAYS read step completion through this method - never through the raw
        truthiness of ``state[key]``. A degraded record (see
        :meth:`mark_degraded`) is a non-empty dict, so ``if state[key]:`` would
        read it as "complete" and re-introduce the very bug it exists to stop.

        A bare truthy value is the legacy shape written before degraded records
        existed; it is honoured as complete, because that is what it claims and
        we cannot retroactively know better.
        """
        value = self.state.get(key)
        if isinstance(value, dict):
            return bool(value.get("complete"))
        return bool(value)

    def mark_degraded(self, key: str, reason: str, **details: Any) -> None:
        """
        Record that a step RAN but could not do the work it exists to do.

        "There was nothing to do" and "we could not do it" used to be written
        identically - as ``True``. So a run on a machine that was missing the
        corpus (or pointed at an empty graph) permanently recorded the step as
        finished, and every later, properly provisioned run printed
        ``[SKIP] ... already created`` and did nothing. The zero-result run
        masqueraded as success, and the skip message made it look intentional.

        The record IS persisted - an operator can read WHY the step is pending -
        but :meth:`is_complete` returns False, so the work is retried.

        Args:
            key: Checkpoint step key.
            reason: Short machine-readable cause, e.g. "corpus_missing".
            **details: Extra context stored alongside (counts, source, ...).
        """
        # `complete` is stripped from details rather than being splatted over: the
        # whole point of this record is that it is NOT complete, and with `**details`
        # last a caller passing `complete=True` as context would silently flip it
        # back to done. That is one careless kwarg away from restoring the exact bug
        # this method exists to prevent, so make it unexpressible.
        details.pop("complete", None)
        self.state[key] = {
            "complete": False,
            "reason": reason,
            "at": datetime.now().isoformat(),
            **details,
        }
        self.save()

    def clear(self, key: str) -> None:
        """Un-mark a completed step so it re-runs (MERGE keeps it idempotent)."""
        self.state[key] = self._defaults().get(key, False)
        self.save()

    def save(self) -> None:
        self.state["last_updated"] = datetime.now().isoformat()
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.checkpoint_file.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(self.state, f, indent=2)
        tmp_path.rename(self.checkpoint_file)

    def mark_complete(self, key: str, value: Any = True) -> None:
        self.state[key] = value
        self.save()

    def reset(self) -> None:
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        self.state = self._load()
        print("Checkpoint reset.")


# ---------------------------------------------------------------------------
# Neo4j Sprint 9 Loader
# ---------------------------------------------------------------------------
class Neo4jSprint9Loader:
    """Async batch loader for Sprint 9 data into Neo4j Aura."""

    def __init__(self) -> None:
        self.driver: Optional[Any] = None
        self.checkpoint = CheckpointManager(CHECKPOINT_FILE)
        self.stats: Dict[str, int] = {
            "dtc": 0,
            "vehicles": 0,
            "engines": 0,
            "complaints": 0,
            "dtc_complaint_rels": 0,
            "dtc_complaint_nodes_created": 0,
            "dtc_nodes_created": 0,
            "vehicle_complaint_rels": 0,
            "vehicle_engine_rels": 0,
        }

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------
    async def connect(self) -> None:
        if not NEO4J_URI or not NEO4J_PASSWORD:
            print("Error: NEO4J_URI and NEO4J_PASSWORD environment variables required.")
            print("  NEO4J_URI=neo4j+s://xxx NEO4J_PASSWORD=xxx python ...")
            sys.exit(1)

        self.driver = AsyncGraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            max_connection_lifetime=300,
            max_connection_pool_size=50,
            connection_acquisition_timeout=60,
        )
        async with self.driver.session() as session:
            result = await session.run("RETURN 1 AS n")
            await result.consume()
        print("[OK] Connected to Neo4j Aura")

    async def close(self) -> None:
        if self.driver:
            await self.driver.close()

    async def _reconnect(self) -> None:
        try:
            await self.close()
        except Exception:
            pass
        await self.connect()

    async def execute_with_retry(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        retries: int = MAX_RETRIES,
    ) -> Any:
        last_error: Optional[Exception] = None
        for attempt in range(retries):
            try:
                async with self.driver.session() as session:
                    result = await session.run(query, params or {})
                    summary = await result.consume()
                    return summary
            except Exception as exc:
                last_error = exc
                if attempt < retries - 1:
                    delay = RETRY_DELAY * (2**attempt)
                    print(
                        f"  [WARN] Retry {attempt + 1}/{retries} "
                        f"after {delay}s: {str(exc)[:80]}"
                    )
                    await asyncio.sleep(delay)
                    await self._reconnect()
        if last_error is not None:
            raise last_error

    # ------------------------------------------------------------------
    # Relationship budget helpers (Aura Free: 400K relationships)
    # ------------------------------------------------------------------
    @staticmethod
    def _counters(summary: Any) -> Tuple[int, int]:
        """(nodes_created, relationships_created) from a Neo4j ResultSummary."""
        counters = getattr(summary, "counters", None)
        if counters is None:
            return 0, 0
        return (
            int(getattr(counters, "nodes_created", 0) or 0),
            int(getattr(counters, "relationships_created", 0) or 0),
        )

    async def count_relationships(self) -> Optional[int]:
        """Total relationship count of the graph, or None if it cannot be read."""
        try:
            async with self.driver.session() as session:
                result = await session.run("MATCH ()-[r]->() RETURN count(r) AS total")
                record = await result.single()
                await result.consume()
                return int(record["total"]) if record else None
        except Exception as exc:
            print(f"  [WARN] Could not read relationship count: {str(exc)[:80]}")
            return None

    async def relationship_headroom(self, reserve: int) -> Tuple[Optional[int], int]:
        """
        Return (current_total, allowed) for a relationship-creating step.

        `allowed` is what is left of AURA_FREE_REL_CAP after the current graph
        and `reserve` (the relationships later steps in this run still need).
        If the count cannot be read we fall back to the configured per-step
        limit rather than assuming an empty graph.
        """
        current = await self.count_relationships()
        if current is None:
            return None, DEFAULT_DTC_REL_LIMIT
        allowed = AURA_FREE_REL_CAP - current - reserve
        used_pct = 100.0 * current / AURA_FREE_REL_CAP
        print(
            f"  Relationship budget: {current:,} / {AURA_FREE_REL_CAP:,} used "
            f"({used_pct:.1f}%), reserve {reserve:,} -> {max(allowed, 0):,} available"
        )
        if current >= AURA_FREE_REL_CAP * REL_WARN_FRACTION:
            print(
                f"  [WARN] Graph is at {used_pct:.1f}% of the Aura Free "
                "relationship cap. Prune data or upgrade the tier."
            )
        return current, max(allowed, 0)

    # ------------------------------------------------------------------
    # Indexes
    # ------------------------------------------------------------------
    async def create_indexes(self) -> None:
        # All IF NOT EXISTS, so this step is safe to re-run on every load.
        #
        # Complaint.make backs VehicleService's common-issues aggregation, which
        # filters `c.make IN $make_variants` on a BARE property precisely so this
        # index can serve it as a seek. Without it that query is a full label
        # scan over every Complaint node (~50K today, and the full-corpus DTC
        # extraction below adds more on every run). The composite mirrors the
        # (v.make, v.model) index above and serves the exact make+model lookups
        # in create_vehicle_complaint_relationships.
        #
        # Index count is not what Aura Free meters (nodes and the 400K
        # relationship cap are), and range indexes on two short string
        # properties are cheap, so these two are effectively free.
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (d:DTC) ON (d.code)",
            "CREATE INDEX IF NOT EXISTS FOR (v:Vehicle) ON (v.make, v.model)",
            "CREATE INDEX IF NOT EXISTS FOR (c:Complaint) ON (c.odi_id)",
            "CREATE INDEX IF NOT EXISTS FOR (c:Complaint) ON (c.make)",
            "CREATE INDEX IF NOT EXISTS FOR (c:Complaint) ON (c.make, c.model)",
            "CREATE INDEX IF NOT EXISTS FOR (e:Engine) ON (e.code)",
        ]
        for idx in indexes:
            try:
                await self.execute_with_retry(idx)
            except Exception as exc:
                print(f"  Index warning: {str(exc)[:80]}")
        print("[OK] Indexes created / verified")

    # ------------------------------------------------------------------
    # 1. DTC Codes
    # ------------------------------------------------------------------
    async def load_dtc_codes(self) -> None:
        if self.checkpoint.is_complete("dtc_loaded"):
            print("[SKIP] DTC codes already loaded")
            return

        dtc_file = DATA_DIR / "dtc_codes" / "all_codes_complete.json"
        if not dtc_file.exists():
            print(f"[WARN] DTC file not found: {dtc_file}")
            return

        with open(dtc_file) as f:
            data = json.load(f)

        all_codes: List[Dict[str, Any]] = data.get("codes", [])
        # The curated file contains a handful of entries that are not DTC codes
        # at all (PEACE, PACED, P93AF, UA80E, UA80F) - they were written by the
        # hex-permissive regex in sync_nhtsa.py. Keep them out of the graph.
        codes = [c for c in all_codes if is_valid_dtc_code(c.get("code"))]
        rejected = [str(c.get("code")) for c in all_codes if not is_valid_dtc_code(c.get("code"))]
        if rejected:
            print(f"  [WARN] Skipping {len(rejected)} malformed DTC codes: {', '.join(rejected)}")
        if not codes:
            print("[WARN] No DTC codes found in file")
            return

        print(f"Loading {len(codes):,} DTC codes ...")

        query = """
        UNWIND $codes AS dtc
        MERGE (d:DTC {code: dtc.code})
        SET d.description_en = dtc.description,
            d.description_hu = dtc.description_hu,
            d.category = dtc.category,
            d.severity = dtc.severity,
            d.system = dtc.system,
            d.is_generic = dtc.is_generic,
            d.sources = dtc.sources,
            d.symptoms = dtc.symptoms,
            d.possible_causes = dtc.possible_causes,
            d.diagnostic_steps = dtc.diagnostic_steps,
            d.related_codes = dtc.related_codes
        """

        with tqdm(total=len(codes), desc="DTC Codes", unit="code") as pbar:
            for i in range(0, len(codes), BATCH_SIZE):
                batch = codes[i : i + BATCH_SIZE]
                batch_params = [
                    {
                        "code": c.get("code", ""),
                        "description": c.get("description", ""),
                        "description_hu": c.get("description_hu"),
                        "category": c.get("category"),
                        "severity": c.get("severity"),
                        "system": c.get("system"),
                        "is_generic": c.get("is_generic", True),
                        "sources": c.get("sources", []),
                        "symptoms": c.get("symptoms", []),
                        "possible_causes": c.get("possible_causes", []),
                        "diagnostic_steps": c.get("diagnostic_steps", []),
                        "related_codes": c.get("related_codes", []),
                    }
                    for c in batch
                ]
                await self.execute_with_retry(query, {"codes": batch_params})
                pbar.update(len(batch))
                self.stats["dtc"] += len(batch)

        self.checkpoint.mark_complete("dtc_loaded")
        print(f"[OK] Loaded {self.stats['dtc']:,} DTC codes")

    # ------------------------------------------------------------------
    # 2. Vehicles
    # ------------------------------------------------------------------
    async def load_vehicles(self) -> None:
        if self.checkpoint.is_complete("vehicles_loaded"):
            print("[SKIP] Vehicles already loaded")
            return

        vehicles_file = DATA_DIR / "vehicles" / "vehicles_master.json"
        if not vehicles_file.exists():
            print(f"[WARN] Vehicles file not found: {vehicles_file}")
            return

        with open(vehicles_file) as f:
            data = json.load(f)

        makes = data.get("makes", [])
        # Filter to makes that actually have models
        makes_with_models = [m for m in makes if m.get("model_count", 0) > 0]

        # Flatten make+model pairs
        vehicle_rows: List[Dict[str, Any]] = []
        for make_data in makes_with_models:
            make_name = make_data.get("make_name", "")
            for model_data in make_data.get("models", []):
                model_name = model_data.get("model_name", "")
                if not model_name:
                    continue
                years = model_data.get("years", [])
                year_start = min(years) if years else None
                year_end = max(years) if years else None
                vehicle_rows.append(
                    {
                        "make": make_name,
                        "model": model_name,
                        "year_start": year_start,
                        "year_end": year_end,
                        "make_id_nhtsa": make_data.get("make_id_nhtsa"),
                        "model_id_nhtsa": model_data.get("model_id_nhtsa"),
                        "sources": model_data.get("sources", []),
                    }
                )

        if not vehicle_rows:
            print("[WARN] No vehicle rows to insert")
            return

        print(
            f"Loading {len(vehicle_rows):,} vehicle models "
            f"from {len(makes_with_models)} makes ..."
        )

        query = """
        UNWIND $vehicles AS v
        MERGE (veh:Vehicle {make: v.make, model: v.model})
        SET veh.year_start = v.year_start,
            veh.year_end = v.year_end,
            veh.make_id_nhtsa = v.make_id_nhtsa,
            veh.model_id_nhtsa = v.model_id_nhtsa,
            veh.sources = v.sources
        """

        with tqdm(total=len(vehicle_rows), desc="Vehicles", unit="veh") as pbar:
            for i in range(0, len(vehicle_rows), BATCH_SIZE):
                batch = vehicle_rows[i : i + BATCH_SIZE]
                await self.execute_with_retry(query, {"vehicles": batch})
                pbar.update(len(batch))
                self.stats["vehicles"] += len(batch)

        self.checkpoint.mark_complete("vehicles_loaded")
        print(f"[OK] Loaded {self.stats['vehicles']:,} vehicles")

    # ------------------------------------------------------------------
    # 3. Engines (EPA)
    # ------------------------------------------------------------------
    async def load_engines(self) -> None:
        if self.checkpoint.is_complete("engines_loaded"):
            print("[SKIP] Engines already loaded")
            return

        engine_file = DATA_DIR / "epa" / "engine_specs.json"
        if not engine_file.exists():
            print(f"[WARN] Engine specs file not found: {engine_file}")
            return

        with open(engine_file) as f:
            raw_records: List[Dict[str, Any]] = json.load(f)

        # Deduplicate to unique engine configurations
        # Key: make + displacement + cylinders + fuel_type + has_turbo + has_supercharger
        seen_engines: Dict[str, Dict[str, Any]] = {}
        for rec in raw_records:
            make = rec.get("make", "UNKNOWN")
            displacement = rec.get("displacement") or 0
            cylinders = rec.get("cylinders") or 0
            fuel_type = (rec.get("fuel_type") or "Unknown").replace(" ", "_")
            has_turbo = rec.get("has_turbo", False)
            has_sc = rec.get("has_supercharger", False)

            suffix_parts = []
            if has_turbo:
                suffix_parts.append("T")
            if has_sc:
                suffix_parts.append("SC")
            suffix = "_".join(suffix_parts) if suffix_parts else "NA"

            engine_code = f"{make}_{displacement}L_{cylinders}cyl_{fuel_type}_{suffix}"

            if engine_code not in seen_engines:
                seen_engines[engine_code] = {
                    "code": engine_code,
                    "name": rec.get("engine", ""),
                    "displacement_l": displacement,
                    "cylinders": cylinders,
                    "fuel_type": rec.get("fuel_type", ""),
                    "fuel_category": rec.get("fuel_category", ""),
                    "manufacturer": make,
                    "has_turbo": has_turbo,
                    "has_supercharger": has_sc,
                }

        engines = list(seen_engines.values())
        print(
            f"Loading {len(engines):,} unique engine configurations "
            f"(from {len(raw_records):,} EPA records) ..."
        )

        query = """
        UNWIND $engines AS e
        MERGE (eng:Engine {code: e.code})
        SET eng.name = e.name,
            eng.displacement_l = e.displacement_l,
            eng.cylinders = e.cylinders,
            eng.fuel_type = e.fuel_type,
            eng.fuel_category = e.fuel_category,
            eng.manufacturer = e.manufacturer,
            eng.has_turbo = e.has_turbo,
            eng.has_supercharger = e.has_supercharger
        """

        with tqdm(total=len(engines), desc="Engines", unit="eng") as pbar:
            for i in range(0, len(engines), BATCH_SIZE):
                batch = engines[i : i + BATCH_SIZE]
                await self.execute_with_retry(query, {"engines": batch})
                pbar.update(len(batch))
                self.stats["engines"] += len(batch)

        self.checkpoint.mark_complete("engines_loaded")
        print(f"[OK] Loaded {self.stats['engines']:,} engine configurations")

    # ------------------------------------------------------------------
    # 4. Complaints (50K, safety-critical first)
    # ------------------------------------------------------------------
    def _safety_score(self, complaint: Dict[str, Any]) -> Tuple[int, ...]:
        """Higher safety score = more critical. Used for descending sort."""
        deaths = int(complaint.get("deaths") or 0)
        injuries = int(complaint.get("injuries") or 0)
        fire = 1 if complaint.get("fire") else 0
        crash = 1 if complaint.get("crash") else 0
        return (deaths, injuries, fire, crash)

    def _collect_complaints_sorted(self) -> List[Dict[str, Any]]:
        """
        Pick which complaints become Complaint NODES.

        This is a NODE-budget decision only: safety-ranked, capped at
        COMPLAINT_LIMIT, preferring the pre-sampled file from
        sample_complaints.py (memory-efficient) over the raw flat files.

        The DTC extraction no longer depends on this selection - it runs over
        the full corpus in scan_corpus_for_dtc_mentions() and MERGEs the extra
        Complaint nodes it needs. Keeping the safety ranking here is fine
        precisely because the two concerns are now decoupled.
        """
        complaints_dir = DATA_DIR / "nhtsa" / "complaints_flat"
        sampled_file = complaints_dir / "sampled_50k_embedding.json"

        # Prefer pre-sampled file (already safety-sorted, ~50K)
        if sampled_file.exists():
            print(f"  Using pre-sampled file: {sampled_file.name}")
            with open(sampled_file) as f:
                complaints = json.load(f)
            # It's a JSON array, not wrapped in {"complaints": [...]}
            if isinstance(complaints, dict):
                complaints = complaints.get("complaints", [])
            print(f"  Loaded {len(complaints):,} pre-sampled complaints")
            # Still respect the limit
            if len(complaints) > COMPLAINT_LIMIT:
                complaints.sort(key=self._safety_score, reverse=True)
                complaints = complaints[:COMPLAINT_LIMIT]
            return complaints

        # Fallback: read raw flat files (WARNING: high memory usage)
        if not complaints_dir.exists():
            print(f"[WARN] Complaints directory not found: {complaints_dir}")
            return []

        print(
            "[WARN] Sampled file not found — reading raw flat files. "
            "Run sample_complaints.py first for lower memory usage."
        )

        all_complaints: List[Dict[str, Any]] = []
        for fname in COMPLAINT_FILES:
            fpath = complaints_dir / fname
            if not fpath.exists():
                print(f"  [WARN] Missing file: {fpath.name}")
                continue
            print(f"  Reading {fpath.name} ...")
            with open(fpath) as f:
                data = json.load(f)
            file_complaints = data.get("complaints", [])
            all_complaints.extend(file_complaints)
            print(f"    -> {len(file_complaints):,} records")

        if not all_complaints:
            return []

        print(f"  Total raw complaints: {len(all_complaints):,}")
        print("  Sorting by safety criticality ...")

        # Sort descending by safety score (deaths, injuries, fire, crash)
        all_complaints.sort(key=self._safety_score, reverse=True)

        # Deduplicate by odi_number (keep first = highest safety score)
        seen_odi: Set[str] = set()
        unique: List[Dict[str, Any]] = []
        for c in all_complaints:
            odi = str(c.get("odi_number", ""))
            if odi and odi in seen_odi:
                continue
            seen_odi.add(odi)
            unique.append(c)
            if len(unique) >= COMPLAINT_LIMIT:
                break

        print(
            f"  Selected {len(unique):,} unique complaints (limit {COMPLAINT_LIMIT:,})"
        )
        return unique

    async def load_complaints(self) -> None:
        if self.checkpoint.is_complete("complaints_loaded"):
            print("[SKIP] Complaints already loaded")
            return

        complaints = self._collect_complaints_sorted()
        if not complaints:
            # No complaint data on this machine is a PROVISIONING problem, not a
            # finished import. Marking it complete (as this used to) makes every
            # later run on a properly provisioned machine print "[SKIP] already
            # loaded" over an empty graph.
            print("[WARN] No complaints to load - step NOT marked complete (data missing)")
            self.checkpoint.mark_degraded(
                "complaints_loaded",
                reason="no_complaint_data",
                source=str(DATA_DIR / "nhtsa" / "complaints_flat"),
            )
            return

        print(f"Loading {len(complaints):,} complaints into Neo4j ...")

        query = """
        UNWIND $complaints AS c
        MERGE (comp:Complaint {odi_id: c.odi_id})
        SET comp.make = c.make,
            comp.model = c.model,
            comp.year = c.year,
            comp.component = c.component,
            comp.description = c.description,
            comp.crash = c.crash,
            comp.fire = c.fire,
            comp.injuries = c.injuries,
            comp.deaths = c.deaths,
            comp.date_received = c.date_received
        """

        with tqdm(total=len(complaints), desc="Complaints", unit="rec") as pbar:
            for i in range(0, len(complaints), BATCH_SIZE):
                batch_raw = complaints[i : i + BATCH_SIZE]
                batch_params = [
                    {
                        "odi_id": str(c.get("odi_number", "")),
                        "make": (c.get("make") or "").upper(),
                        "model": (c.get("model") or "").upper(),
                        "year": int(c.get("model_year") or 0),
                        "component": c.get("component", ""),
                        "description": (c.get("summary") or "")[:5000],
                        "crash": bool(c.get("crash")),
                        "fire": bool(c.get("fire")),
                        "injuries": int(c.get("injuries") or 0),
                        "deaths": int(c.get("deaths") or 0),
                        "date_received": c.get("date_received", ""),
                    }
                    for c in batch_raw
                ]
                await self.execute_with_retry(query, {"complaints": batch_params})
                pbar.update(len(batch_raw))
                self.stats["complaints"] += len(batch_raw)

                # Checkpoint every 10 batches
                if (i // BATCH_SIZE) % 10 == 0:
                    self.checkpoint.state["complaints_current_index"] = i + len(
                        batch_raw
                    )
                    self.checkpoint.save()

        self.checkpoint.mark_complete("complaints_loaded")
        print(f"[OK] Loaded {self.stats['complaints']:,} complaints")

    # ------------------------------------------------------------------
    # 5. Relationships
    # ------------------------------------------------------------------
    async def _scan_graph_for_dtc_mentions(self) -> DtcScanResult:
        """
        Fallback scan: extract DTC codes from the Complaint nodes already in
        the graph. Only used when the raw corpus files are unavailable - it can
        never see more than the COMPLAINT_LIMIT safety-ranked sample, which is
        exactly the limitation this step is trying to escape.
        """
        fetch_query = """
        MATCH (c:Complaint)
        WHERE c.description IS NOT NULL AND c.description <> ''
        RETURN c.odi_id AS odi_id, c.description AS description,
               c.date_received AS date_received
        """
        async with self.driver.session() as session:
            result = await session.run(fetch_query)
            records = await result.data()

        print(f"  Scanning {len(records):,} in-graph complaint descriptions ...")
        pairs: List[Dict[str, Any]] = []
        matched: Set[str] = set()
        for rec in records:
            odi_id = rec.get("odi_id") or ""
            if not odi_id:
                continue
            for code in extract_dtc_codes(rec.get("description")):
                pairs.append(
                    {
                        "odi_id": odi_id,
                        "code": code,
                        "date_received": rec.get("date_received") or "",
                    }
                )
                matched.add(odi_id)
        stats = {
            "complaints_scanned": len(records),
            "complaints_with_dtc": len(matched),
            "mentions": len(pairs),
            "unique_codes": len({p["code"] for p in pairs}),
            "hit_rate": (len(matched) / len(records)) if records else 0.0,
            "truncated": False,
            "source": "neo4j-nodes",
        }
        # No node payloads: these complaints are already in the graph.
        return DtcScanResult(pairs=pairs, complaint_nodes={}, stats=stats)

    async def create_dtc_complaint_relationships(
        self, rel_limit: int = DEFAULT_DTC_REL_LIMIT
    ) -> None:
        """
        Extract DTC codes from complaint narratives and link them.

        Three changes vs. the original implementation:
          1. the extraction runs over the FULL flat-file corpus (~1.66M
             complaints), not over the 50K safety-ranked node sample;
          2. the DTC node is MERGEd, not MATCHed, so a real code that is not in
             the curated 6.8K set no longer silently drops its edge (such nodes
             carry source='complaint_extraction' + is_curated=false);
          3. every write is budgeted against the Aura Free 400K relationship
             cap and capped by `rel_limit`.

        Complaint nodes created here carry source='dtc_extraction' and are
        excluded from the later HAS_COMPLAINT step - VehicleService's
        common-issues query filters on the Complaint's own make/model/year
        properties, so those edges would cost budget without adding reach.
        """
        if self.checkpoint.is_complete("dtc_complaint_rels"):
            print("[SKIP] DTC-Complaint relationships already created")
            print("       (use --redo-dtc-rels to re-run the improved extraction)")
            return

        print("Creating DTC <-> Complaint relationships ...")

        complaints_dir = DATA_DIR / "nhtsa" / "complaints_flat"
        available = [f for f in COMPLAINT_FILES if (complaints_dir / f).exists()]
        # THE completion predicate for this step. The whole point of the step is
        # to extract from the FULL ~1.66M-record corpus; the in-graph fallback
        # can never see more than the COMPLAINT_LIMIT safety-ranked sample, which
        # is precisely the limitation this step exists to escape (that sample is
        # what produced the ~107 MENTIONS_DTC edges the live graph has). A run
        # without the corpus therefore did NOT do this step's work, however many
        # edges it happened to write, and must not close it for later runs.
        corpus_present = bool(available)
        if corpus_present:
            scan = scan_corpus_for_dtc_mentions(complaints_dir, available)
            scan.stats["source"] = "flat-files"
        else:
            print(
                "  [WARN] No raw corpus files in "
                f"{complaints_dir} - falling back to the in-graph sample. "
                "Expect a fraction of the possible edges."
            )
            scan = await self._scan_graph_for_dtc_mentions()

        print_scan_report(scan)

        if not scan.pairs:
            if corpus_present:
                # The real corpus really does contain no DTC mention. The step
                # ran over everything it was supposed to; an empty answer here is
                # a RESULT, not a failure.
                print("  No DTC codes found in complaint narratives (full corpus scanned)")
                self.checkpoint.mark_complete("dtc_complaint_rels")
            else:
                print(
                    "  [WARN] No DTC codes found AND no corpus was available - "
                    "this run proved nothing. Step NOT marked complete; re-run "
                    "it where the flat-file corpus is present."
                )
                self.checkpoint.mark_degraded(
                    "dtc_complaint_rels",
                    reason="corpus_missing",
                    source=scan.stats.get("source", "unknown"),
                    complaints_scanned=scan.stats.get("complaints_scanned", 0),
                    mentions=0,
                    created=0,
                )
            return

        # --- Budget --------------------------------------------------------
        _current, allowed = await self.relationship_headroom(REL_RESERVE_FOR_LATER_STEPS)
        effective_limit = min(rel_limit, allowed)
        if effective_limit <= 0:
            print(
                "  [WARN] No relationship headroom left under the "
                f"{AURA_FREE_REL_CAP:,} cap - skipping DTC edge creation. "
                "Nothing was written."
            )
            return

        curated = load_curated_dtc_codes()
        rel_pairs = prioritize_dtc_pairs(scan.pairs, curated, effective_limit)
        if len(rel_pairs) < len(scan.pairs):
            print(
                f"  [WARN] Capped at {len(rel_pairs):,} of {len(scan.pairs):,} "
                "mentions (curated codes and newest complaints first)."
            )

        non_curated = sorted({p["code"] for p in rel_pairs if p["code"] not in curated})
        print(
            f"  Writing {len(rel_pairs):,} mentions; "
            f"{len(non_curated):,} of the codes are not in the curated set "
            "and will be created as extraction-sourced DTC nodes"
        )

        # Attach the Complaint node payload so complaints outside the
        # safety-ranked node set can be MERGEd on the fly.
        batch_rows: List[Dict[str, Any]] = []
        for pair in rel_pairs:
            node = scan.complaint_nodes.get(pair["odi_id"], {})
            batch_rows.append(
                {
                    "odi_id": pair["odi_id"],
                    "code": pair["code"],
                    "category": dtc_category(pair["code"]),
                    "make": node.get("make", ""),
                    "model": node.get("model", ""),
                    "year": node.get("year", 0),
                    "component": node.get("component", ""),
                    "description": node.get("description", ""),
                    "crash": node.get("crash", False),
                    "fire": node.get("fire", False),
                    "injuries": node.get("injuries", 0),
                    "deaths": node.get("deaths", 0),
                    "date_received": node.get("date_received", ""),
                }
            )

        # MERGE everywhere => re-running the step is idempotent.
        # ON CREATE only, so nodes loaded by load_complaints()/load_dtc_codes()
        # keep their richer curated properties.
        rel_query = """
        UNWIND $pairs AS p
        MERGE (c:Complaint {odi_id: p.odi_id})
        ON CREATE SET c.make = p.make,
                      c.model = p.model,
                      c.year = p.year,
                      c.component = p.component,
                      c.description = p.description,
                      c.crash = p.crash,
                      c.fire = p.fire,
                      c.injuries = p.injuries,
                      c.deaths = p.deaths,
                      c.date_received = p.date_received,
                      c.source = $complaint_source
        MERGE (d:DTC {code: p.code})
        ON CREATE SET d.source = $dtc_source,
                      d.is_curated = false,
                      d.category = p.category,
                      d.description_en = 'Extracted from NHTSA complaint text'
        MERGE (c)-[r:MENTIONS_DTC]->(d)
        ON CREATE SET r.source = 'complaint_text'
        """

        created_rels = 0
        created_nodes = 0
        with tqdm(total=len(batch_rows), desc="DTC-Complaint Rels", unit="rel") as pbar:
            for i in range(0, len(batch_rows), BATCH_SIZE):
                batch = batch_rows[i : i + BATCH_SIZE]
                summary = await self.execute_with_retry(
                    rel_query,
                    {
                        "pairs": batch,
                        "complaint_source": COMPLAINT_SOURCE_EXTRACTION,
                        "dtc_source": DTC_SOURCE_EXTRACTION,
                    },
                )
                nodes_added, rels_added = self._counters(summary)
                created_nodes += nodes_added
                created_rels += rels_added
                pbar.update(len(batch))
                # Running budget log every ~25 batches (12.5K rows).
                if (i // BATCH_SIZE) % 25 == 0 and created_rels:
                    print(f"    +{created_rels:,} new MENTIONS_DTC so far")

        self.stats["dtc_complaint_rels"] = created_rels
        self.stats["dtc_complaint_nodes_created"] = created_nodes
        self.checkpoint.state["dtc_complaint_rels_created"] = created_rels
        if corpus_present:
            self.checkpoint.mark_complete("dtc_complaint_rels")
        else:
            # Edges WERE written, but only from the safety-ranked in-graph
            # sample. Closing the step here would let a partial result block the
            # full extraction on the next properly provisioned run.
            print(
                "  [WARN] These edges came from the in-graph sample, not the "
                "full corpus - step left OPEN so a corpus-equipped run redoes it."
            )
            self.checkpoint.mark_degraded(
                "dtc_complaint_rels",
                reason="corpus_missing",
                source=scan.stats.get("source", "unknown"),
                complaints_scanned=scan.stats.get("complaints_scanned", 0),
                mentions=len(scan.pairs),
                created=created_rels,
            )
        print(
            f"[OK] {created_rels:,} new MENTIONS_DTC relationships "
            f"({len(batch_rows):,} mentions processed, the rest already existed), "
            f"{created_nodes:,} nodes created"
        )

    async def create_vehicle_complaint_relationships(self) -> None:
        """Link complaints to vehicles by make+model (batched)."""
        if self.checkpoint.is_complete("vehicle_complaint_rels"):
            print("[SKIP] Vehicle-Complaint relationships already created")
            return

        print("Creating Vehicle <-> Complaint relationships ...")

        # Budget check: this step runs LAST and is the first casualty of the
        # Aura Free cap, so refuse to start it if there is no headroom.
        _current, allowed = await self.relationship_headroom(0)
        if allowed <= 0:
            print(
                "  [WARN] No relationship headroom left under the "
                f"{AURA_FREE_REL_CAP:,} cap - skipping HAS_COMPLAINT creation."
            )
            return

        # First, collect unique (make, model) pairs from complaints.
        # Extraction-sourced complaints are excluded: the common-issues query
        # reads their make/model/year properties directly, so a HAS_COMPLAINT
        # edge would only consume budget.
        collect_query = """
        MATCH (c:Complaint)
        WHERE c.make IS NOT NULL AND c.model IS NOT NULL
          AND (c.source IS NULL OR c.source <> $extraction_source)
        RETURN DISTINCT c.make AS make, c.model AS model
        """

        async with self.driver.session() as session:
            result = await session.run(
                collect_query, {"extraction_source": COMPLAINT_SOURCE_EXTRACTION}
            )
            pairs = [{"make": rec["make"], "model": rec["model"]} async for rec in result]

        print(f"  Found {len(pairs):,} unique (make, model) pairs to link")

        # Batch-create relationships using UNWIND
        rel_query = """
        UNWIND $pairs AS p
        MATCH (c:Complaint)
        WHERE toUpper(c.make) = toUpper(p.make)
          AND toUpper(c.model) = toUpper(p.model)
          AND (c.source IS NULL OR c.source <> $extraction_source)
        MATCH (v:Vehicle)
        WHERE toUpper(v.make) = toUpper(p.make)
          AND toUpper(v.model) = toUpper(p.model)
        MERGE (v)-[:HAS_COMPLAINT]->(c)
        """

        total_cnt = 0
        batch_size = 50  # Small batches for Aura free tier

        with tqdm(total=len(pairs), desc="Vehicle-Complaint Rels", unit="pair") as pbar:
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i : i + batch_size]
                # execute_with_retry returns a ResultSummary (not records), so
                # the created count comes from the write counters. The previous
                # `result_data[0].get("cnt")` raised TypeError on every batch.
                summary = await self.execute_with_retry(
                    rel_query,
                    {"pairs": batch, "extraction_source": COMPLAINT_SOURCE_EXTRACTION},
                )
                _nodes, rels_added = self._counters(summary)
                total_cnt += rels_added
                if total_cnt >= allowed:
                    print(
                        f"  [WARN] Stopping: {total_cnt:,} HAS_COMPLAINT edges "
                        f"created, headroom ({allowed:,}) exhausted."
                    )
                    break
                pbar.update(len(batch))

        self.stats["vehicle_complaint_rels"] = total_cnt
        self.checkpoint.mark_complete("vehicle_complaint_rels")
        print(f"[OK] Created {total_cnt:,} Vehicle-Complaint relationships")

    async def create_vehicle_engine_relationships(self) -> None:
        """Link vehicles to engines via EPA engine_specs data."""
        if self.checkpoint.is_complete("vehicle_engine_rels"):
            print("[SKIP] Vehicle-Engine relationships already created")
            return

        engine_file = DATA_DIR / "epa" / "engine_specs.json"
        if not engine_file.exists():
            print(
                "[WARN] Engine specs file not found - Vehicle-Engine rels NOT "
                "marked complete (data missing, not done)"
            )
            self.checkpoint.mark_degraded(
                "vehicle_engine_rels",
                reason="engine_specs_missing",
                source=str(engine_file),
            )
            return

        print("Creating Vehicle <-> Engine relationships ...")

        with open(engine_file) as f:
            raw_records: List[Dict[str, Any]] = json.load(f)

        # Build unique (make, model, engine_code) tuples
        seen: Set[Tuple[str, str, str]] = set()
        rel_rows: List[Dict[str, str]] = []

        for rec in raw_records:
            make = rec.get("make", "UNKNOWN")
            model = rec.get("model", "")
            displacement = rec.get("displacement") or 0
            cylinders = rec.get("cylinders") or 0
            fuel_type = (rec.get("fuel_type") or "Unknown").replace(" ", "_")
            has_turbo = rec.get("has_turbo", False)
            has_sc = rec.get("has_supercharger", False)

            suffix_parts = []
            if has_turbo:
                suffix_parts.append("T")
            if has_sc:
                suffix_parts.append("SC")
            suffix = "_".join(suffix_parts) if suffix_parts else "NA"

            engine_code = f"{make}_{displacement}L_{cylinders}cyl_{fuel_type}_{suffix}"
            key = (make, model, engine_code)
            if key not in seen:
                seen.add(key)
                rel_rows.append(
                    {"make": make, "model": model, "engine_code": engine_code}
                )

        print(f"  {len(rel_rows):,} unique (Vehicle, Engine) pairs to link")

        rel_query = """
        UNWIND $rows AS r
        MATCH (v:Vehicle)
        WHERE toUpper(v.make) = toUpper(r.make)
          AND toUpper(v.model) = toUpper(r.model)
        MATCH (e:Engine {code: r.engine_code})
        MERGE (v)-[:USES_ENGINE]->(e)
        """

        with tqdm(total=len(rel_rows), desc="Vehicle-Engine Rels", unit="rel") as pbar:
            for i in range(0, len(rel_rows), BATCH_SIZE):
                batch = rel_rows[i : i + BATCH_SIZE]
                await self.execute_with_retry(rel_query, {"rows": batch})
                pbar.update(len(batch))
                self.stats["vehicle_engine_rels"] += len(batch)

        self.checkpoint.mark_complete("vehicle_engine_rels")
        print(
            f"[OK] Created {self.stats['vehicle_engine_rels']:,} "
            "Vehicle-Engine relationships"
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    async def reset_sprint9_data(self) -> None:
        """
        Remove Sprint 9 specific data. DESTRUCTIVE - explicit --reset only.

        Also removes the nodes the DTC extraction created (they are tagged with
        source properties, so curated DTC nodes and the safety-ranked complaint
        sample are never touched).
        """
        print("Resetting Sprint 9 data ...")
        queries = [
            "MATCH ()-[r:MENTIONS_DTC]->() DELETE r",
            "MATCH ()-[r:USES_ENGINE]->() DELETE r",
            "MATCH (e:Engine) DETACH DELETE e",
            f"MATCH (c:Complaint) WHERE c.source = '{COMPLAINT_SOURCE_EXTRACTION}' "
            "DETACH DELETE c",
            f"MATCH (d:DTC) WHERE d.source = '{DTC_SOURCE_EXTRACTION}' DETACH DELETE d",
        ]
        for q in queries:
            try:
                await self.execute_with_retry(q)
                print(f"  Done: {q[:60]}...")
            except Exception as exc:
                print(f"  Error: {str(exc)[:80]}")

        self.checkpoint.reset()
        print("[OK] Sprint 9 data reset. Re-run with --all to reload.")

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    async def run(
        self,
        do_dtc: bool = False,
        do_vehicles: bool = False,
        do_engines: bool = False,
        do_complaints: bool = False,
        do_all: bool = False,
        do_reset: bool = False,
        redo_dtc_rels: bool = False,
        dtc_rel_limit: int = DEFAULT_DTC_REL_LIMIT,
    ) -> None:
        start_time = time.time()
        print("=" * 64)
        print("  SPRINT 9 NEO4J DATA SYNC")
        print("=" * 64)
        print(f"  Timestamp : {datetime.now().isoformat()}")
        print(f"  Neo4j URI : {NEO4J_URI[:40]}...")
        print(f"  Batch size: {BATCH_SIZE}")
        if self.checkpoint.state["last_updated"]:
            print(f"  Last run  : {self.checkpoint.state['last_updated']}")
        print()

        try:
            await self.connect()

            if do_reset:
                await self.reset_sprint9_data()
                return

            if redo_dtc_rels:
                # Non-destructive: the step re-runs with MERGE semantics, so
                # existing edges are kept and only missing ones are added.
                self.checkpoint.clear("dtc_complaint_rels")
                print("  --redo-dtc-rels: DTC extraction step will re-run")

            await self.create_indexes()

            if do_all or do_dtc:
                await self.load_dtc_codes()

            if do_all or do_vehicles:
                await self.load_vehicles()

            if do_all or do_engines:
                await self.load_engines()

            if do_all or do_complaints:
                await self.load_complaints()

            # Relationships — only when running --all or when the
            # prerequisite node types have been loaded in this or
            # previous runs
            if do_all or do_complaints or redo_dtc_rels:
                await self.create_dtc_complaint_relationships(rel_limit=dtc_rel_limit)

            if do_all or do_complaints:
                await self.create_vehicle_complaint_relationships()

            if do_all or do_engines:
                await self.create_vehicle_engine_relationships()

            elapsed = time.time() - start_time
            print()
            print("=" * 64)
            print("  SYNC COMPLETE")
            print("=" * 64)
            for label, count in self.stats.items():
                if count > 0:
                    print(f"  {label:<30s}: {count:>10,}")
            print(f"  {'elapsed':<30s}: {elapsed:>10.1f}s")
            print()

        except Exception as exc:
            print(f"\n[ERROR] {exc}")
            print("Progress saved to checkpoint. Re-run to resume.")
            raise
        finally:
            await self.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sprint 9 Neo4j Data Sync",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Load everything (DTC + Vehicles + Engines + Complaints + Rels)",
    )
    parser.add_argument("--dtc", action="store_true", help="Load DTC codes only")
    parser.add_argument(
        "--vehicles", action="store_true", help="Load vehicle makes/models only"
    )
    parser.add_argument("--engines", action="store_true", help="Load EPA engines only")
    parser.add_argument(
        "--complaints",
        action="store_true",
        help=(
            f"Load top {COMPLAINT_LIMIT:,} complaint NODES (safety-critical first) "
            "+ run the full-corpus DTC extraction and the vehicle links"
        ),
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help=(
            "DESTRUCTIVE: remove Sprint 9 specific data (Engine nodes, "
            "MENTIONS_DTC/USES_ENGINE rels, extraction-created DTC/Complaint nodes)"
        ),
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help=(
            "Scan the local complaint corpus for DTC codes and print what WOULD "
            "be created. No database connection, no credentials, no writes."
        ),
    )
    parser.add_argument(
        "--redo-dtc-rels",
        action="store_true",
        help=(
            "Re-run the DTC extraction step even if the checkpoint says it is "
            "done (idempotent: MERGE only adds missing edges)"
        ),
    )
    parser.add_argument(
        "--dtc-rel-limit",
        type=int,
        default=DEFAULT_DTC_REL_LIMIT,
        help=(
            f"Max MENTIONS_DTC relationships to create in one run "
            f"(default {DEFAULT_DTC_REL_LIMIT:,}; also bounded by the "
            f"{AURA_FREE_REL_CAP:,} Aura Free cap)"
        ),
    )
    return parser.parse_args()


def run_scan_only() -> int:
    """DB-free dry run of the corpus DTC extraction. Returns a process exit code."""
    complaints_dir = DATA_DIR / "nhtsa" / "complaints_flat"
    available = [f for f in COMPLAINT_FILES if (complaints_dir / f).exists()]
    if not available:
        print(f"[ERROR] No complaint corpus files found in {complaints_dir}")
        print(f"        Expected any of: {', '.join(COMPLAINT_FILES)}")
        return 1
    scan = scan_corpus_for_dtc_mentions(complaints_dir, available)
    scan.stats["source"] = "flat-files"
    print_scan_report(scan)
    curated = load_curated_dtc_codes()
    would_write = len(prioritize_dtc_pairs(scan.pairs, curated, DEFAULT_DTC_REL_LIMIT))
    print(
        f"  With --dtc-rel-limit {DEFAULT_DTC_REL_LIMIT:,} this run would write "
        f"{would_write:,} MENTIONS_DTC relationships and up to "
        f"{len(scan.complaint_nodes):,} extraction-sourced Complaint nodes."
    )
    print("  (No database was contacted.)")
    return 0


async def main() -> None:
    args = parse_args()

    if args.scan_only:
        sys.exit(run_scan_only())

    # Default to --all if nothing specified
    if not any(
        [
            args.all,
            args.dtc,
            args.vehicles,
            args.engines,
            args.complaints,
            args.reset,
            args.redo_dtc_rels,
        ]
    ):
        print("No flags specified, defaulting to --all")
        args.all = True

    loader = Neo4jSprint9Loader()
    await loader.run(
        do_dtc=args.dtc,
        do_vehicles=args.vehicles,
        do_engines=args.engines,
        do_complaints=args.complaints,
        do_all=args.all,
        do_reset=args.reset,
        redo_dtc_rels=args.redo_dtc_rels,
        dtc_rel_limit=args.dtc_rel_limit,
    )


if __name__ == "__main__":
    asyncio.run(main())
