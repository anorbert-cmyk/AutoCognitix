#!/usr/bin/env python3
"""
Intelligent NHTSA Complaint Sampling for AutoCognitix Sprint 9.

Selects ~200K most diagnostically important complaints from 1.66M total
using a priority-based sampling strategy:

  1. Safety-critical (crash/fire/injuries/deaths) -- ALL kept
  2. DTC-relevant (summary mentions P/B/C/U + 4 digits) -- ALL kept
  3. Top 30 makes with component diversity -- max 500 per (make, component)
  4. Recent years priority -- 2020+: 100%, 2015-2019: 50%, older: 20%

Two-pass streaming approach to avoid loading 1.8GB into memory:
  Pass 1: Classify every complaint, collect odi_numbers per category
  Pass 2: Re-read files, extract only selected complaints, write output

Outputs:
  - sampled_200k.json        -- Full sampled dataset (for PostgreSQL)
  - sampled_50k_embedding.json -- Top 50K for Qdrant embedding
  - sampling_stats.json      -- Detailed statistics

Usage:
    python scripts/sample_complaints.py
    python scripts/sample_complaints.py --target 200000 --seed 42
    python scripts/sample_complaints.py --target 150000 --embedding-target 30000
"""

import argparse
import json
import logging
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "nhtsa" / "complaints_flat"
CHECKPOINT_DIR = PROJECT_ROOT / "scripts" / "checkpoints"

SOURCE_FILES: List[str] = [
    "2000-2004.json",
    "2005-2009.json",
    "2010-2014.json",
    "2015-2019.json",
    "2020-2024.json",
    "2025-2026.json",
]

DTC_PATTERN = re.compile(r"\b[PBCU][0-9]{4}\b")

TOP_30_MAKES: Set[str] = {
    "FORD",
    "CHEVROLET",
    "TOYOTA",
    "HONDA",
    "JEEP",
    "DODGE",
    "NISSAN",
    "HYUNDAI",
    "KIA",
    "CHRYSLER",
    "GMC",
    "VOLKSWAGEN",
    "BMW",
    "SUBARU",
    "RAM",
    "PONTIAC",
    "MAZDA",
    "TESLA",
    "BUICK",
    "MERCEDES-BENZ",
    "SATURN",
    "CADILLAC",
    "ACURA",
    "MERCURY",
    "AUDI",
    "LEXUS",
    "LINCOLN",
    "MERCEDES BENZ",
    "VOLVO",
    "MITSUBISHI",
}

MAX_PER_MAKE_COMPONENT = 500

YEAR_SAMPLING_RATES: List[Tuple[int, int, float]] = [
    (2020, 2026, 1.0),
    (2015, 2019, 0.5),
    (2000, 2014, 0.2),
]

CHUNK_SIZE = 1024 * 1024  # 1 MB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Streaming JSON parser
# ---------------------------------------------------------------------------


def _find_complaints_array(f) -> Optional[str]:
    """Read file chunks until we locate the 'complaints' JSON array start.

    Returns the remaining buffer positioned right after the opening bracket,
    or None if the array was not found.
    """
    buffer = ""
    marker = '"complaints"'

    while True:
        chunk = f.read(CHUNK_SIZE)
        if not chunk:
            return None
        buffer += chunk

        idx = buffer.find(marker)
        if idx == -1:
            if len(buffer) > len(marker) + 10:
                buffer = buffer[-(len(marker) + 10) :]
            continue

        bracket_idx = buffer.find("[", idx + len(marker))
        if bracket_idx == -1:
            buffer = buffer[idx:]
            continue

        return buffer[bracket_idx + 1 :]


class _JsonObjectExtractor:
    """State machine that extracts JSON objects from a streaming character buffer.

    Tracks brace depth and string escaping to find complete top-level JSON
    objects inside a JSON array.  Call :meth:`feed` with successive buffer
    slices; complete objects are returned via the ``objects`` list.
    """

    __slots__ = ("depth", "escape_next", "in_string", "obj_start", "objects")

    def __init__(self) -> None:
        self.depth = 0
        self.obj_start = -1
        self.in_string = False
        self.escape_next = False
        self.objects: List[Dict[str, Any]] = []

    def scan(self, buffer: str, start: int) -> Tuple[int, bool]:
        """Scan *buffer* starting at *start*.

        Returns ``(new_pos, array_ended)`` where *new_pos* is the scan
        position after processing and *array_ended* is True when the
        closing ``]`` of the array is reached.
        """
        pos = start
        while pos < len(buffer):
            ch = buffer[pos]

            if self.escape_next:
                self.escape_next = False
                pos += 1
                continue

            if self.in_string:
                if ch == "\\":
                    self.escape_next = True
                elif ch == '"':
                    self.in_string = False
                pos += 1
                continue

            # Outside a string
            if ch == '"':
                self.in_string = True
            elif ch == "{":
                if self.depth == 0:
                    self.obj_start = pos
                self.depth += 1
            elif ch == "}":
                self.depth -= 1
                if self.depth == 0 and self.obj_start != -1:
                    self._emit_object(buffer[self.obj_start : pos + 1])
                    self.obj_start = -1
            elif ch == "]" and self.depth == 0:
                return pos, True

            pos += 1
        return pos, False

    def _emit_object(self, obj_str: str) -> None:
        try:
            self.objects.append(json.loads(obj_str))
        except json.JSONDecodeError:
            logger.warning(
                "Failed to parse complaint object: %.100s...",
                obj_str[:100],
            )


def _extract_objects_from_buffer(
    buffer: str,
    f,
) -> Iterator[Dict[str, Any]]:
    """Parse individual JSON objects from a streaming buffer.

    Reads more chunks as needed and yields complete complaint dicts.
    """
    extractor = _JsonObjectExtractor()
    pos = 0

    while True:
        pos, array_ended = extractor.scan(buffer, pos)

        # Yield any objects found during this scan
        if extractor.objects:
            yield from extractor.objects
            extractor.objects.clear()

        if array_ended:
            return

        # Compact buffer and read more data
        if extractor.obj_start > 0:
            buffer = buffer[extractor.obj_start :]
            pos -= extractor.obj_start
            extractor.obj_start = 0
        elif extractor.obj_start == -1:
            buffer = ""
            pos = 0

        chunk = f.read(CHUNK_SIZE)
        if not chunk:
            return
        buffer += chunk


def stream_complaints(filepath: Path) -> Iterator[Dict[str, Any]]:
    """Yield complaint dicts one at a time from a JSON file.

    The file has the structure ``{"metadata": {...}, "complaints": [...]}``.
    Uses buffered reading with a brace-depth state machine so that at most
    one complaint object is in memory at a time.
    """
    with filepath.open(encoding="utf-8") as f:
        buffer = _find_complaints_array(f)
        if buffer is None:
            return
        yield from _extract_objects_from_buffer(buffer, f)


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------


def is_safety_critical(complaint: Dict[str, Any]) -> bool:
    """Returns True if the complaint involves crash, fire, injury, or death."""
    return bool(
        complaint.get("crash")
        or complaint.get("fire")
        or (complaint.get("injuries") or 0) > 0
        or (complaint.get("deaths") or 0) > 0
    )


def has_dtc_code(complaint: Dict[str, Any]) -> bool:
    """Returns True if the complaint summary contains a DTC code pattern."""
    summary = complaint.get("summary") or ""
    return bool(DTC_PATTERN.search(summary))


def get_year_sampling_rate(model_year: Optional[int]) -> float:
    """Returns the sampling rate for a given model year."""
    if model_year is None:
        return 0.2
    for year_min, year_max, rate in YEAR_SAMPLING_RATES:
        if year_min <= model_year <= year_max:
            return rate
    return 0.2


# ---------------------------------------------------------------------------
# Checkpoint support
# ---------------------------------------------------------------------------


def load_checkpoint(checkpoint_path: Path) -> Optional[Dict[str, Any]]:
    """Load checkpoint from disk if it exists."""
    if checkpoint_path.exists():
        try:
            with checkpoint_path.open(encoding="utf-8") as f:
                data = json.load(f)
            logger.info("Loaded checkpoint from %s", checkpoint_path)
            return data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load checkpoint: %s", e)
    return None


def save_checkpoint(checkpoint_path: Path, data: Dict[str, Any]) -> None:
    """Save checkpoint to disk atomically."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f)
    tmp_path.rename(checkpoint_path)
    logger.info("Saved checkpoint to %s", checkpoint_path)


# ---------------------------------------------------------------------------
# Pass 1: Classify all complaints
# ---------------------------------------------------------------------------


def _restore_pass1_checkpoint(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """Restore full pass-1 result from a completed checkpoint."""
    return {
        "selected_ids": set(checkpoint["selected_ids"]),
        "safety_ids": set(checkpoint["safety_ids"]),
        "dtc_ids": set(checkpoint["dtc_ids"]),
        "embedding_ids": set(checkpoint["embedding_ids"]),
        "stats": checkpoint["stats"],
    }


def _scan_all_files(
    data_dir: Path,
    checkpoint: Optional[Dict[str, Any]],
    checkpoint_path: Path,
) -> Tuple[
    Set[str],
    Set[str],
    Dict[str, List[str]],
    List[Tuple[str, int]],
    int,
    Set[str],
    float,
]:
    """Scan source files and classify each complaint.

    Returns (safety_ids, dtc_ids, make_component_ids, general_pool,
             total_scanned, seen_odi, elapsed).
    """
    safety_ids: Set[str] = set()
    dtc_ids: Set[str] = set()
    make_component_ids: Dict[str, List[str]] = defaultdict(list)
    general_pool: List[Tuple[str, int]] = []
    total_scanned = 0
    seen_odi: Set[str] = set()
    completed_files: Set[str] = set()

    if checkpoint:
        completed_files = set(checkpoint.get("completed_files", []))
        safety_ids = set(checkpoint.get("safety_ids", []))
        dtc_ids = set(checkpoint.get("dtc_ids", []))
        total_scanned = checkpoint.get("total_scanned", 0)
        logger.info(
            "Resuming from checkpoint: %d files done, %d scanned",
            len(completed_files),
            total_scanned,
        )

    start_time = time.time()

    for filename in SOURCE_FILES:
        if filename in completed_files:
            logger.info("  Skipping %s (already processed)", filename)
            continue

        filepath = data_dir / filename
        if not filepath.exists():
            logger.warning("  File not found: %s (skipping)", filepath)
            continue

        file_count, file_safety, file_dtc, file_dupes = 0, 0, 0, 0
        logger.info("  Scanning: %s", filename)

        for complaint in stream_complaints(filepath):
            odi = complaint.get("odi_number", "")
            if not odi or odi in seen_odi:
                file_dupes += 1 if odi else 0
                continue
            seen_odi.add(odi)
            total_scanned += 1
            file_count += 1

            if is_safety_critical(complaint):
                safety_ids.add(odi)
                file_safety += 1
            if has_dtc_code(complaint):
                dtc_ids.add(odi)
                file_dtc += 1

            make = (complaint.get("make") or "").upper()
            component = (
                complaint.get("component") or complaint.get("components") or "UNKNOWN"
            )
            model_year = complaint.get("model_year")

            if make in TOP_30_MAKES:
                make_component_ids[f"{make}||{component}"].append(odi)

            if odi not in safety_ids and odi not in dtc_ids:
                general_pool.append((odi, model_year if model_year else 0))

            if file_count % 100000 == 0:
                logger.info("    ... %d scanned in %s", file_count, filename)

        logger.info(
            "  Done %s: %d unique, %d safety, %d DTC, %d dupes",
            filename,
            file_count,
            file_safety,
            file_dtc,
            file_dupes,
        )
        completed_files.add(filename)

        save_checkpoint(
            checkpoint_path,
            {
                "pass1_complete": False,
                "completed_files": list(completed_files),
                "safety_ids": list(safety_ids),
                "dtc_ids": list(dtc_ids),
                "total_scanned": total_scanned,
            },
        )

    elapsed = time.time() - start_time
    return (
        safety_ids,
        dtc_ids,
        make_component_ids,
        general_pool,
        total_scanned,
        seen_odi,
        elapsed,
    )


def _apply_selection(
    safety_ids: Set[str],
    dtc_ids: Set[str],
    make_component_ids: Dict[str, List[str]],
    general_pool: List[Tuple[str, int]],
    rng: random.Random,
) -> Tuple[Set[str], Set[str], int, int]:
    """Apply the priority-based selection strategy.

    Returns (selected_ids, embedding_ids, diversity_added, year_added).
    """
    selected_ids: Set[str] = set()

    # Step 1: All safety-critical
    selected_ids.update(safety_ids)
    logger.info("  Step 1 - Safety-critical: %d selected", len(safety_ids))

    # Step 2: All DTC-relevant
    dtc_new = len(dtc_ids - selected_ids)
    selected_ids.update(dtc_ids)
    logger.info("  Step 2 - DTC-relevant: %d total (%d new)", len(dtc_ids), dtc_new)

    # Step 3: Component diversity (capped per make+component)
    diversity_added = 0
    for _key, odi_list in make_component_ids.items():
        remaining = [oid for oid in odi_list if oid not in selected_ids]
        if not remaining:
            continue
        if len(remaining) > MAX_PER_MAKE_COMPONENT:
            rng.shuffle(remaining)
            remaining = remaining[:MAX_PER_MAKE_COMPONENT]
        selected_ids.update(remaining)
        diversity_added += len(remaining)
    logger.info("  Step 3 - Make/component diversity: %d new", diversity_added)

    # Step 4: Year-based sampling
    year_added = 0
    for odi, year in general_pool:
        if odi in selected_ids:
            continue
        rate = get_year_sampling_rate(year if year else None)
        if rng.random() < rate:
            selected_ids.add(odi)
            year_added += 1
    logger.info("  Step 4 - Year-based sampling: %d new", year_added)
    logger.info("  TOTAL SELECTED: %d", len(selected_ids))

    # Embedding base: safety + DTC
    embedding_ids = (safety_ids | dtc_ids) & selected_ids
    logger.info("  Embedding base (safety + DTC): %d", len(embedding_ids))

    return selected_ids, embedding_ids, diversity_added, year_added


def pass1_classify(
    data_dir: Path,
    rng: random.Random,
    checkpoint_path: Path,
) -> Dict[str, Any]:
    """Scan all source files, classify, and select odi_numbers to extract."""
    logger.info("=" * 70)
    logger.info("PASS 1: Classifying all complaints")
    logger.info("=" * 70)

    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint and checkpoint.get("pass1_complete"):
        logger.info("Pass 1 already completed (from checkpoint), skipping scan")
        return _restore_pass1_checkpoint(checkpoint)

    (
        safety_ids,
        dtc_ids,
        make_component_ids,
        general_pool,
        total_scanned,
        seen_odi,
        elapsed,
    ) = _scan_all_files(data_dir, checkpoint, checkpoint_path)

    logger.info(
        "Pass 1 scan complete: %d unique in %.1fs (%.0f/s)",
        total_scanned,
        elapsed,
        total_scanned / max(elapsed, 0.001),
    )

    selected_ids, embedding_ids, diversity_added, year_added = _apply_selection(
        safety_ids,
        dtc_ids,
        make_component_ids,
        general_pool,
        rng,
    )

    stats = {
        "total_scanned": total_scanned,
        "unique_after_dedup": len(seen_odi),
        "safety_critical_count": len(safety_ids),
        "dtc_relevant_count": len(dtc_ids),
        "overlap_safety_dtc": len(safety_ids & dtc_ids),
        "diversity_added": diversity_added,
        "year_sampling_added": year_added,
        "total_selected": len(selected_ids),
        "embedding_base_count": len(embedding_ids),
        "pass1_elapsed_seconds": round(elapsed, 1),
    }

    save_checkpoint(
        checkpoint_path,
        {
            "pass1_complete": True,
            "selected_ids": list(selected_ids),
            "safety_ids": list(safety_ids),
            "dtc_ids": list(dtc_ids),
            "embedding_ids": list(embedding_ids),
            "stats": stats,
        },
    )

    return {
        "selected_ids": selected_ids,
        "safety_ids": safety_ids,
        "dtc_ids": dtc_ids,
        "embedding_ids": embedding_ids,
        "stats": stats,
    }


# ---------------------------------------------------------------------------
# Pass 2: Extract selected complaints and write output files
# ---------------------------------------------------------------------------


def _extract_from_files(
    data_dir: Path,
    selected_ids: Set[str],
    safety_ids: Set[str],
    dtc_ids: Set[str],
    embedding_ids: Set[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], float]:
    """Read source files and collect selected complaints.

    Returns (extracted, embedding_pool, extraction_stats, elapsed).
    """
    extracted: List[Dict[str, Any]] = []
    embedding_pool: List[Dict[str, Any]] = []
    remaining = set(selected_ids)

    by_make: Dict[str, int] = defaultdict(int)
    by_year: Dict[str, int] = defaultdict(int)
    by_component: Dict[str, int] = defaultdict(int)
    safety_extracted = 0
    dtc_extracted = 0

    start_time = time.time()

    for filename in SOURCE_FILES:
        filepath = data_dir / filename
        if not filepath.exists():
            logger.warning("  File not found: %s (skipping)", filepath)
            continue

        file_extracted = 0
        logger.info("  Reading: %s", filename)

        for complaint in stream_complaints(filepath):
            odi = complaint.get("odi_number", "")
            if odi not in remaining:
                continue

            remaining.discard(odi)
            extracted.append(complaint)
            file_extracted += 1

            by_make[(complaint.get("make") or "UNKNOWN").upper()] += 1
            by_year[str(complaint.get("model_year") or "UNKNOWN")] += 1
            by_component[complaint.get("component") or "UNKNOWN"] += 1

            if odi in safety_ids:
                safety_extracted += 1
            if odi in dtc_ids:
                dtc_extracted += 1
            if odi in embedding_ids:
                embedding_pool.append(complaint)

            if file_extracted % 50000 == 0:
                logger.info(
                    "    ... %d extracted from %s (%d remaining)",
                    file_extracted,
                    filename,
                    len(remaining),
                )

        logger.info("  Extracted %d from %s", file_extracted, filename)
        if not remaining:
            logger.info("  All selected found, skipping remaining files")
            break

    elapsed = time.time() - start_time

    if remaining:
        logger.warning("  %d odi_numbers not found in any file!", len(remaining))

    extraction_stats = {
        "safety_extracted": safety_extracted,
        "dtc_extracted": dtc_extracted,
        "missing_ids": len(remaining),
        "by_make_top30": dict(
            sorted(by_make.items(), key=lambda x: x[1], reverse=True)[:30]
        ),
        "by_year": dict(sorted(by_year.items())),
        "by_component_top25": dict(
            sorted(by_component.items(), key=lambda x: x[1], reverse=True)[:25]
        ),
    }
    return extracted, embedding_pool, extraction_stats, elapsed


def _build_embedding_subset(
    embedding_pool: List[Dict[str, Any]],
    extracted: List[Dict[str, Any]],
    safety_ids: Set[str],
    dtc_ids: Set[str],
    embedding_target: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Build the final embedding subset, targeting ``embedding_target`` items."""
    if len(embedding_pool) < embedding_target:
        embed_odi = {c.get("odi_number") for c in embedding_pool}
        extras = [c for c in extracted if c.get("odi_number") not in embed_odi]
        rng.shuffle(extras)
        embedding_pool.extend(extras[: embedding_target - len(embedding_pool)])

    if len(embedding_pool) > embedding_target:
        priority = [
            c
            for c in embedding_pool
            if c.get("odi_number") in safety_ids or c.get("odi_number") in dtc_ids
        ]
        other = [
            c
            for c in embedding_pool
            if c.get("odi_number") not in safety_ids
            and c.get("odi_number") not in dtc_ids
        ]
        rng.shuffle(other)
        slots = max(0, embedding_target - len(priority))
        embedding_pool = priority + other[:slots]

    return embedding_pool


def _write_json_streaming(filepath: Path, complaints: List[Dict[str, Any]]) -> None:
    """Write complaint dicts as a JSON array, streaming to disk."""
    with filepath.open("w", encoding="utf-8") as f:
        f.write("[\n")
        for i, complaint in enumerate(complaints):
            if i > 0:
                f.write(",\n")
            json.dump(complaint, f, ensure_ascii=False)
        f.write("\n]\n")


def pass2_extract(
    data_dir: Path,
    output_dir: Path,
    selected_ids: Set[str],
    safety_ids: Set[str],
    dtc_ids: Set[str],
    embedding_ids: Set[str],
    embedding_target: int,
    rng: random.Random,
) -> Dict[str, Any]:
    """Re-read source files, extract selected complaints, write output."""
    logger.info("=" * 70)
    logger.info("PASS 2: Extracting %d selected complaints", len(selected_ids))
    logger.info("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)

    extracted, embedding_pool, ext_stats, elapsed = _extract_from_files(
        data_dir,
        selected_ids,
        safety_ids,
        dtc_ids,
        embedding_ids,
    )

    logger.info("Pass 2 extraction: %d complaints in %.1fs", len(extracted), elapsed)

    embedding_pool = _build_embedding_subset(
        embedding_pool,
        extracted,
        safety_ids,
        dtc_ids,
        embedding_target,
        rng,
    )
    logger.info("Embedding subset: %d complaints", len(embedding_pool))

    main_path = output_dir / "sampled_200k.json"
    embed_path = output_dir / "sampled_50k_embedding.json"

    logger.info("Writing main output: %s", main_path)
    _write_json_streaming(main_path, extracted)
    main_mb = round(main_path.stat().st_size / (1024 * 1024), 1)

    logger.info("Writing embedding output: %s", embed_path)
    _write_json_streaming(embed_path, embedding_pool)
    embed_mb = round(embed_path.stat().st_size / (1024 * 1024), 1)

    logger.info("Output sizes: main=%.1f MB, embedding=%.1f MB", main_mb, embed_mb)

    return {
        "extracted_count": len(extracted),
        "embedding_count": len(embedding_pool),
        "main_output_size_mb": main_mb,
        "embedding_output_size_mb": embed_mb,
        "pass2_elapsed_seconds": round(elapsed, 1),
        **ext_stats,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Intelligent NHTSA Complaint Sampling for AutoCognitix",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=200000,
        help="Target complaint count for main sample (default: 200000)",
    )
    parser.add_argument(
        "--embedding-target",
        type=int,
        default=50000,
        help="Target count for embedding subset (default: 50000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable checkpoint loading (fresh start)",
    )
    return parser.parse_args()


def _log_summary(
    pass1_stats: Dict[str, Any],
    pass2_stats: Dict[str, Any],
    output_dir: Path,
    total_elapsed: float,
) -> None:
    """Print a final summary of the sampling run."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("SAMPLING COMPLETE")
    logger.info("=" * 70)
    logger.info("  Total scanned:    %d", pass1_stats["total_scanned"])
    logger.info("  Safety-critical:  %d", pass1_stats["safety_critical_count"])
    logger.info("  DTC-relevant:     %d", pass1_stats["dtc_relevant_count"])
    logger.info("  Total selected:   %d", pass2_stats["extracted_count"])
    logger.info("  Embedding subset: %d", pass2_stats["embedding_count"])
    logger.info("  Total time:       %.1f seconds", total_elapsed)
    logger.info("")
    logger.info("Output files:")
    logger.info(
        "  Main:      %s (%.1f MB)",
        output_dir / "sampled_200k.json",
        pass2_stats["main_output_size_mb"],
    )
    logger.info(
        "  Embedding: %s (%.1f MB)",
        output_dir / "sampled_50k_embedding.json",
        pass2_stats["embedding_output_size_mb"],
    )
    logger.info("  Stats:     %s", output_dir / "sampling_stats.json")


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    output_dir = Path(args.output_dir) if args.output_dir else data_dir
    checkpoint_path = CHECKPOINT_DIR / "sample_complaints_checkpoint.json"

    if args.no_checkpoint and checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Removed existing checkpoint (--no-checkpoint)")

    rng = random.Random(args.seed)

    logger.info("AutoCognitix NHTSA Complaint Sampler")
    logger.info("  Data dir:         %s", data_dir)
    logger.info("  Output dir:       %s", output_dir)
    logger.info("  Target count:     %d", args.target)
    logger.info("  Embedding target: %d", args.embedding_target)
    logger.info("  Random seed:      %d", args.seed)
    logger.info("")

    found_files = 0
    for filename in SOURCE_FILES:
        filepath = data_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            logger.info("  Found: %s (%.0f MB)", filename, size_mb)
            found_files += 1
        else:
            logger.warning("  MISSING: %s", filename)

    if found_files == 0:
        logger.error("No source files found in %s -- aborting", data_dir)
        sys.exit(1)

    total_start = time.time()

    result = pass1_classify(data_dir, rng, checkpoint_path)
    pass1_stats = result["stats"]

    pass2_stats = pass2_extract(
        data_dir=data_dir,
        output_dir=output_dir,
        selected_ids=result["selected_ids"],
        safety_ids=result["safety_ids"],
        dtc_ids=result["dtc_ids"],
        embedding_ids=result["embedding_ids"],
        embedding_target=args.embedding_target,
        rng=rng,
    )

    total_elapsed = time.time() - total_start

    stats = {
        "sampling_config": {
            "target_count": args.target,
            "embedding_target": args.embedding_target,
            "random_seed": args.seed,
            "max_per_make_component": MAX_PER_MAKE_COMPONENT,
            "year_sampling_rates": {
                f"{y_min}-{y_max}": rate for y_min, y_max, rate in YEAR_SAMPLING_RATES
            },
            "top_30_makes": sorted(TOP_30_MAKES),
        },
        "pass1_classification": pass1_stats,
        "pass2_extraction": pass2_stats,
        "performance": {
            "total_elapsed_seconds": round(total_elapsed, 1),
            "pass1_seconds": pass1_stats.get("pass1_elapsed_seconds", 0),
            "pass2_seconds": pass2_stats.get("pass2_elapsed_seconds", 0),
        },
    }

    stats_path = output_dir / "sampling_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info("Statistics written to: %s", stats_path)

    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Cleaned up checkpoint file")

    _log_summary(pass1_stats, pass2_stats, output_dir, total_elapsed)


if __name__ == "__main__":
    main()
