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
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from neo4j import AsyncGraphDatabase
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

BATCH_SIZE = 500
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds, exponentially increased
COMPLAINT_LIMIT = 50_000

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
CHECKPOINT_FILE = SCRIPT_DIR / "checkpoints" / "neo4j_sprint9.json"

DTC_PATTERN = re.compile(r"\b([PBCU][0-9]{4})\b")

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
# Checkpoint Manager
# ---------------------------------------------------------------------------
class CheckpointManager:
    """Persist progress across runs so we can resume after failures."""

    def __init__(self, checkpoint_file: Path) -> None:
        self.checkpoint_file = checkpoint_file
        self.state: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file) as f:
                return json.load(f)
        return {
            "dtc_loaded": False,
            "vehicles_loaded": False,
            "engines_loaded": False,
            "complaints_loaded": False,
            "complaints_files_done": [],
            "complaints_current_file": None,
            "complaints_current_index": 0,
            "dtc_complaint_rels": False,
            "vehicle_complaint_rels": False,
            "vehicle_engine_rels": False,
            "last_updated": None,
        }

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
        raise last_error  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Indexes
    # ------------------------------------------------------------------
    async def create_indexes(self) -> None:
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (d:DTC) ON (d.code)",
            "CREATE INDEX IF NOT EXISTS FOR (v:Vehicle) ON (v.make, v.model)",
            "CREATE INDEX IF NOT EXISTS FOR (c:Complaint) ON (c.odi_id)",
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
        if self.checkpoint.state["dtc_loaded"]:
            print("[SKIP] DTC codes already loaded")
            return

        dtc_file = DATA_DIR / "dtc_codes" / "all_codes_complete.json"
        if not dtc_file.exists():
            print(f"[WARN] DTC file not found: {dtc_file}")
            return

        with open(dtc_file) as f:
            data = json.load(f)

        codes: List[Dict[str, Any]] = data.get("codes", [])
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
        if self.checkpoint.state["vehicles_loaded"]:
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
        if self.checkpoint.state["engines_loaded"]:
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
        Load complaints for Neo4j — prefers the pre-sampled file from
        sample_complaints.py (memory-efficient) over raw flat files.

        Falls back to raw files only if the sampled file doesn't exist.
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
        if self.checkpoint.state["complaints_loaded"]:
            print("[SKIP] Complaints already loaded")
            return

        complaints = self._collect_complaints_sorted()
        if not complaints:
            print("[WARN] No complaints to load")
            self.checkpoint.mark_complete("complaints_loaded")
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
    async def create_dtc_complaint_relationships(self) -> None:
        """Extract DTC codes from complaint summaries and link them."""
        if self.checkpoint.state["dtc_complaint_rels"]:
            print("[SKIP] DTC-Complaint relationships already created")
            return

        print("Creating DTC <-> Complaint relationships ...")

        # Fetch complaints with descriptions that may contain DTC codes
        fetch_query = """
        MATCH (c:Complaint)
        WHERE c.description IS NOT NULL AND c.description <> ''
        RETURN c.odi_id AS odi_id, c.description AS description
        """

        odi_to_dtcs: Dict[str, List[str]] = {}
        async with self.driver.session() as session:
            result = await session.run(fetch_query)
            records = await result.data()

        print(f"  Scanning {len(records):,} complaint descriptions for DTC codes ...")
        for rec in records:
            description = rec.get("description", "")
            odi_id = rec.get("odi_id", "")
            found_codes = DTC_PATTERN.findall(description)
            if found_codes:
                odi_to_dtcs[odi_id] = list(set(found_codes))

        if not odi_to_dtcs:
            print("  No DTC codes found in complaint descriptions")
            self.checkpoint.mark_complete("dtc_complaint_rels")
            return

        # Flatten to list of (odi_id, code) pairs
        rel_pairs: List[Dict[str, str]] = []
        for odi_id, codes in odi_to_dtcs.items():
            for code in codes:
                rel_pairs.append({"odi_id": odi_id, "code": code})

        print(
            f"  Found {len(rel_pairs):,} DTC mentions "
            f"across {len(odi_to_dtcs):,} complaints"
        )

        rel_query = """
        UNWIND $pairs AS p
        MATCH (c:Complaint {odi_id: p.odi_id})
        MATCH (d:DTC {code: p.code})
        MERGE (c)-[:MENTIONS_DTC]->(d)
        """

        with tqdm(total=len(rel_pairs), desc="DTC-Complaint Rels", unit="rel") as pbar:
            for i in range(0, len(rel_pairs), BATCH_SIZE):
                batch = rel_pairs[i : i + BATCH_SIZE]
                await self.execute_with_retry(rel_query, {"pairs": batch})
                pbar.update(len(batch))
                self.stats["dtc_complaint_rels"] += len(batch)

        self.checkpoint.mark_complete("dtc_complaint_rels")
        print(
            f"[OK] Created {self.stats['dtc_complaint_rels']:,} "
            "DTC-Complaint relationships"
        )

    async def create_vehicle_complaint_relationships(self) -> None:
        """Link complaints to vehicles by make+model (batched)."""
        if self.checkpoint.state["vehicle_complaint_rels"]:
            print("[SKIP] Vehicle-Complaint relationships already created")
            return

        print("Creating Vehicle <-> Complaint relationships ...")

        # First, collect unique (make, model) pairs from complaints
        collect_query = """
        MATCH (c:Complaint)
        WHERE c.make IS NOT NULL AND c.model IS NOT NULL
        RETURN DISTINCT c.make AS make, c.model AS model
        """

        async with self.driver.session() as session:
            result = await session.run(collect_query)
            pairs = [
                {"make": rec["make"], "model": rec["model"]}
                async for rec in result
            ]

        print(f"  Found {len(pairs):,} unique (make, model) pairs to link")

        # Batch-create relationships using UNWIND
        rel_query = """
        UNWIND $pairs AS p
        MATCH (c:Complaint)
        WHERE toUpper(c.make) = toUpper(p.make)
          AND toUpper(c.model) = toUpper(p.model)
        MATCH (v:Vehicle)
        WHERE toUpper(v.make) = toUpper(p.make)
          AND toUpper(v.model) = toUpper(p.model)
        MERGE (v)-[:HAS_COMPLAINT]->(c)
        RETURN count(*) AS cnt
        """

        total_cnt = 0
        batch_size = 50  # Small batches for Aura free tier

        with tqdm(total=len(pairs), desc="Vehicle-Complaint Rels", unit="pair") as pbar:
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i : i + batch_size]
                result_data = await self.execute_with_retry(
                    rel_query, {"pairs": batch}
                )
                if result_data and len(result_data) > 0:
                    total_cnt += result_data[0].get("cnt", 0)
                pbar.update(len(batch))

        self.stats["vehicle_complaint_rels"] = total_cnt
        self.checkpoint.mark_complete("vehicle_complaint_rels")
        print(f"[OK] Created {total_cnt:,} Vehicle-Complaint relationships")

    async def create_vehicle_engine_relationships(self) -> None:
        """Link vehicles to engines via EPA engine_specs data."""
        if self.checkpoint.state["vehicle_engine_rels"]:
            print("[SKIP] Vehicle-Engine relationships already created")
            return

        engine_file = DATA_DIR / "epa" / "engine_specs.json"
        if not engine_file.exists():
            print("[WARN] Engine specs file not found, skipping Vehicle-Engine rels")
            self.checkpoint.mark_complete("vehicle_engine_rels")
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
        """Remove Sprint 9 specific data. Use with caution."""
        print("Resetting Sprint 9 data ...")
        queries = [
            "MATCH ()-[r:MENTIONS_DTC]->() DELETE r",
            "MATCH ()-[r:USES_ENGINE]->() DELETE r",
            "MATCH (e:Engine) DETACH DELETE e",
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
            if do_all or do_complaints:
                await self.create_dtc_complaint_relationships()
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
        help=f"Load top {COMPLAINT_LIMIT:,} complaints (safety-critical first)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Remove Sprint 9 specific data (Engine nodes, MENTIONS_DTC/USES_ENGINE rels)",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    # Default to --all if nothing specified
    if not any(
        [args.all, args.dtc, args.vehicles, args.engines, args.complaints, args.reset]
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
    )


if __name__ == "__main__":
    asyncio.run(main())
