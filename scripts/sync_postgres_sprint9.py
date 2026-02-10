#!/usr/bin/env python3
"""
Sprint 9 PostgreSQL Data Sync.

Loads DTC codes, vehicle makes/models, complaints, and EPA vehicles
into PostgreSQL using raw SQL with psycopg2 for maximum performance.

Usage:
    DATABASE_URL=postgresql://... python scripts/sync_postgres_sprint9.py --all
    DATABASE_URL=postgresql://... python scripts/sync_postgres_sprint9.py --dtc
    DATABASE_URL=postgresql://... python scripts/sync_postgres_sprint9.py --makes
    DATABASE_URL=postgresql://... python scripts/sync_postgres_sprint9.py --complaints
    DATABASE_URL=postgresql://... python scripts/sync_postgres_sprint9.py --epa
    DATABASE_URL=postgresql://... python scripts/sync_postgres_sprint9.py --reset
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import psycopg2
import psycopg2.extras
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
CHECKPOINT_FILE = SCRIPT_DIR / "checkpoints" / "postgres_sprint9.json"

# Data source paths
DTC_FILE = DATA_DIR / "dtc_codes" / "all_codes_complete.json"
VEHICLES_FILE = DATA_DIR / "vehicles" / "vehicles_master.json"
EPA_FILE = DATA_DIR / "epa" / "vehicles_complete.json"
COMPLAINTS_DIR = DATA_DIR / "nhtsa" / "complaints_flat"
SAMPLED_COMPLAINTS_FILE = COMPLAINTS_DIR / "sampled_200k.json"

# Complaint flat files in chronological order
COMPLAINT_FLAT_FILES = [
    COMPLAINTS_DIR / "2000-2004.json",
    COMPLAINTS_DIR / "2005-2009.json",
    COMPLAINTS_DIR / "2010-2014.json",
    COMPLAINTS_DIR / "2015-2019.json",
    COMPLAINTS_DIR / "2020-2024.json",
    COMPLAINTS_DIR / "2025-2026.json",
]

# Batch sizes
BATCH_DTC = 500
BATCH_MAKES = 100
BATCH_MODELS = 500
BATCH_COMPLAINTS = 1000
BATCH_EPA = 1000

# Max complaints to load if no sampled file exists
MAX_COMPLAINTS = 200_000

# DTC extraction regex
DTC_REGEX = re.compile(r"\b[PBCU][0-9]{4}\b")

# ---------------------------------------------------------------------------
# Country mapping for common makes
# ---------------------------------------------------------------------------
MAKE_COUNTRY_MAP: Dict[str, str] = {
    "acura": "Japan",
    "alfa romeo": "Italy",
    "aston martin": "United Kingdom",
    "audi": "Germany",
    "bentley": "United Kingdom",
    "bmw": "Germany",
    "buick": "United States",
    "cadillac": "United States",
    "chevrolet": "United States",
    "chrysler": "United States",
    "citroen": "France",
    "dacia": "Romania",
    "daewoo": "South Korea",
    "daihatsu": "Japan",
    "dodge": "United States",
    "ferrari": "Italy",
    "fiat": "Italy",
    "fisker": "United States",
    "ford": "United States",
    "genesis": "South Korea",
    "gmc": "United States",
    "honda": "Japan",
    "hummer": "United States",
    "hyundai": "South Korea",
    "infiniti": "Japan",
    "isuzu": "Japan",
    "jaguar": "United Kingdom",
    "jeep": "United States",
    "kia": "South Korea",
    "lamborghini": "Italy",
    "land rover": "United Kingdom",
    "lexus": "Japan",
    "lincoln": "United States",
    "lotus": "United Kingdom",
    "lucid": "United States",
    "maserati": "Italy",
    "mazda": "Japan",
    "mclaren": "United Kingdom",
    "mercedes-benz": "Germany",
    "mercedes benz": "Germany",
    "mercury": "United States",
    "mini": "United Kingdom",
    "mitsubishi": "Japan",
    "nissan": "Japan",
    "oldsmobile": "United States",
    "opel": "Germany",
    "peugeot": "France",
    "plymouth": "United States",
    "polestar": "Sweden",
    "pontiac": "United States",
    "porsche": "Germany",
    "ram": "United States",
    "renault": "France",
    "rivian": "United States",
    "rolls-royce": "United Kingdom",
    "saab": "Sweden",
    "saturn": "United States",
    "scion": "Japan",
    "seat": "Spain",
    "skoda": "Czech Republic",
    "smart": "Germany",
    "subaru": "Japan",
    "suzuki": "Japan",
    "tesla": "United States",
    "toyota": "Japan",
    "volkswagen": "Germany",
    "volvo": "Sweden",
    "winnebago": "United States",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def slugify(name: str) -> str:
    """Convert a name to a slug: lowercase, underscores, no special chars."""
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = slug.strip("_")
    return slug


def parse_date_str(date_str: Optional[str]) -> Optional[date]:
    """Parse 'YYYY-MM-DD' string to date, or return None."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def extract_dtc_codes(text: Optional[str]) -> List[str]:
    """Extract DTC codes from a text string using regex."""
    if not text:
        return []
    return sorted(set(DTC_REGEX.findall(text.upper())))


def get_country(make_name: str) -> Optional[str]:
    """Look up country for a vehicle make."""
    key = make_name.lower().strip()
    return MAKE_COUNTRY_MAP.get(key)


def ensure_list(val: Any) -> List[str]:
    """Ensure a value is a list of strings (handles None)."""
    if val is None:
        return []
    if isinstance(val, list):
        return [str(v) for v in val if v]
    return []


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------
def load_checkpoint() -> Dict[str, Any]:
    """Load checkpoint from disk."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {
        "dtc_done": False,
        "makes_done": False,
        "models_done": False,
        "complaints_done": False,
        "epa_done": False,
        "complaints_loaded": 0,
        "complaints_file_idx": 0,
    }


def save_checkpoint(ckpt: Dict[str, Any]) -> None:
    """Save checkpoint to disk (atomic write via tmp+rename)."""
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = CHECKPOINT_FILE.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(ckpt, f, indent=2)
    tmp_path.rename(CHECKPOINT_FILE)


def reset_checkpoint() -> None:
    """Remove checkpoint file."""
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        print("Checkpoint reset.")
    else:
        print("No checkpoint file to reset.")


# ---------------------------------------------------------------------------
# Database connection
# ---------------------------------------------------------------------------
def get_connection() -> psycopg2.extensions.connection:
    """Create a psycopg2 connection from DATABASE_URL."""
    db_url = os.getenv("DATABASE_URL", "")
    if not db_url:
        print("Error: DATABASE_URL environment variable required")
        print(
            "Usage: DATABASE_URL=postgresql://user:pass@host:port/db "
            "python sync_postgres_sprint9.py"
        )
        sys.exit(1)

    # Convert asyncpg URL format to psycopg2
    db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")
    return psycopg2.connect(db_url)


# ---------------------------------------------------------------------------
# DTC Codes sync
# ---------------------------------------------------------------------------
def sync_dtc_codes(conn: psycopg2.extensions.connection) -> int:
    """Load DTC codes into dtc_codes table. Returns count inserted/updated."""
    print("\n=== DTC Codes ===")
    if not DTC_FILE.exists():
        print(f"  SKIP: {DTC_FILE} not found")
        return 0

    with open(DTC_FILE, "r") as f:
        data = json.load(f)

    codes = data.get("codes", [])
    total = len(codes)
    print(f"  Source: {DTC_FILE.name} ({total} codes)")

    sql = """
        INSERT INTO dtc_codes (
            code, description_en, description_hu, category, severity,
            is_generic, system, symptoms, possible_causes,
            diagnostic_steps, related_codes, sources
        )
        VALUES %s
        ON CONFLICT (code) DO UPDATE SET
            description_en = EXCLUDED.description_en,
            description_hu = COALESCE(EXCLUDED.description_hu, dtc_codes.description_hu),
            category = EXCLUDED.category,
            severity = EXCLUDED.severity,
            is_generic = EXCLUDED.is_generic,
            system = COALESCE(EXCLUDED.system, dtc_codes.system),
            symptoms = EXCLUDED.symptoms,
            possible_causes = EXCLUDED.possible_causes,
            diagnostic_steps = EXCLUDED.diagnostic_steps,
            related_codes = EXCLUDED.related_codes,
            sources = EXCLUDED.sources,
            updated_at = NOW()
    """

    upserted = 0
    cur = conn.cursor()

    for i in tqdm(range(0, total, BATCH_DTC), desc="  DTC codes", unit="batch"):
        batch = codes[i : i + BATCH_DTC]
        values = []
        for c in batch:
            code = c.get("code", "").upper().strip()
            if not code:
                continue
            description_en = c.get("description") or c.get("description_en") or ""
            description_hu = c.get("description_hu")
            category = c.get("category", "powertrain")
            severity = c.get("severity", "medium")
            is_generic = c.get("is_generic", True)
            system = c.get("system")
            symptoms = ensure_list(c.get("symptoms"))
            possible_causes = ensure_list(c.get("possible_causes"))
            diagnostic_steps = ensure_list(c.get("diagnostic_steps"))
            related_codes = ensure_list(c.get("related_codes"))
            sources = ensure_list(c.get("sources"))

            values.append(
                (
                    code,
                    description_en,
                    description_hu,
                    category,
                    severity,
                    is_generic,
                    system,
                    symptoms,
                    possible_causes,
                    diagnostic_steps,
                    related_codes,
                    sources,
                )
            )

        if values:
            psycopg2.extras.execute_values(cur, sql, values, page_size=BATCH_DTC)
            upserted += len(values)

    conn.commit()
    cur.close()
    print(f"  Done: {upserted} DTC codes upserted")
    return upserted


# ---------------------------------------------------------------------------
# Vehicle Makes sync
# ---------------------------------------------------------------------------
def sync_vehicle_makes(
    conn: psycopg2.extensions.connection,
) -> Tuple[int, List[Dict[str, Any]]]:
    """Load vehicle makes into vehicle_makes table.

    Returns (count, makes_with_models) where makes_with_models is the
    filtered list of makes that have at least one model.
    """
    print("\n=== Vehicle Makes ===")
    if not VEHICLES_FILE.exists():
        print(f"  SKIP: {VEHICLES_FILE} not found")
        return 0, []

    with open(VEHICLES_FILE, "r") as f:
        data = json.load(f)

    all_makes = data.get("makes", [])
    # Filter: only makes that have at least one model
    makes_with_models = [m for m in all_makes if m.get("model_count", 0) > 0]
    total = len(makes_with_models)
    print(
        f"  Source: {VEHICLES_FILE.name} ({total} makes with models, "
        f"{len(all_makes)} total)"
    )

    sql = """
        INSERT INTO vehicle_makes (id, name, country, nhtsa_make_id)
        VALUES %s
        ON CONFLICT (id) DO UPDATE SET
            name = EXCLUDED.name,
            country = COALESCE(EXCLUDED.country, vehicle_makes.country),
            nhtsa_make_id = COALESCE(EXCLUDED.nhtsa_make_id, vehicle_makes.nhtsa_make_id)
    """

    upserted = 0
    cur = conn.cursor()

    for i in tqdm(range(0, total, BATCH_MAKES), desc="  Makes", unit="batch"):
        batch = makes_with_models[i : i + BATCH_MAKES]
        values = []
        for m in batch:
            make_name = m.get("make_name", "").strip()
            if not make_name:
                continue
            make_id = slugify(make_name)
            if not make_id or len(make_id) > 50:
                make_id = make_id[:50]
            if not make_id:
                continue
            country = get_country(make_name)
            nhtsa_make_id = m.get("make_id_nhtsa")
            values.append((make_id, make_name, country, nhtsa_make_id))

        if values:
            psycopg2.extras.execute_values(cur, sql, values, page_size=BATCH_MAKES)
            upserted += len(values)

    conn.commit()
    cur.close()
    print(f"  Done: {upserted} makes upserted")
    return upserted, makes_with_models


# ---------------------------------------------------------------------------
# Vehicle Models sync
# ---------------------------------------------------------------------------
def sync_vehicle_models(
    conn: psycopg2.extensions.connection,
    makes_with_models: List[Dict[str, Any]],
) -> int:
    """Load vehicle models into vehicle_models table. Returns count."""
    print("\n=== Vehicle Models ===")
    if not makes_with_models:
        print("  SKIP: No makes with models")
        return 0

    # Build list of all models with their make reference
    all_models: List[Tuple[str, str, str, int, Optional[int]]] = []
    for make in makes_with_models:
        make_name = make.get("make_name", "").strip()
        make_slug = slugify(make_name)
        if not make_slug:
            continue
        make_slug = make_slug[:50]

        for model in make.get("models", []):
            model_name = model.get("model_name", "").strip()
            if not model_name:
                continue
            model_slug = slugify(model_name)
            if not model_slug:
                continue
            model_id = f"{make_slug}_{model_slug}"
            if len(model_id) > 50:
                model_id = model_id[:50]

            years = model.get("years", [])
            years_int = [int(y) for y in years if y is not None]
            if years_int:
                year_start = min(years_int)
                year_end = max(years_int)
            else:
                # No year data; skip model
                continue

            all_models.append(
                (
                    model_id,
                    model_name,
                    make_slug,
                    year_start,
                    year_end,
                )
            )

    # Deduplicate by model_id (first element of tuple) — keep latest year range
    deduped: Dict[str, Tuple[str, str, str, int, int]] = {}
    for model_tuple in all_models:
        mid = model_tuple[0]
        if mid in deduped:
            # Merge year ranges
            existing = deduped[mid]
            deduped[mid] = (
                mid,
                model_tuple[1],  # name
                model_tuple[2],  # make_id
                min(existing[3], model_tuple[3]),  # year_start
                max(existing[4], model_tuple[4]),  # year_end
            )
        else:
            deduped[mid] = model_tuple
    all_models = list(deduped.values())

    total = len(all_models)
    print(f"  Total models to sync: {total}")

    sql = """
        INSERT INTO vehicle_models (id, name, make_id, year_start, year_end)
        VALUES %s
        ON CONFLICT (id) DO UPDATE SET
            name = EXCLUDED.name,
            make_id = EXCLUDED.make_id,
            year_start = LEAST(EXCLUDED.year_start, vehicle_models.year_start),
            year_end = GREATEST(EXCLUDED.year_end, vehicle_models.year_end)
    """

    upserted = 0
    cur = conn.cursor()

    for i in tqdm(range(0, total, BATCH_MODELS), desc="  Models", unit="batch"):
        batch = all_models[i : i + BATCH_MODELS]
        if batch:
            psycopg2.extras.execute_values(cur, sql, batch, page_size=BATCH_MODELS)
            upserted += len(batch)

    conn.commit()
    cur.close()
    print(f"  Done: {upserted} models upserted")
    return upserted


# ---------------------------------------------------------------------------
# Complaints sync
# ---------------------------------------------------------------------------
def _load_complaints_from_flat_files(max_count: int) -> List[Dict[str, Any]]:
    """Load complaints from flat files, up to max_count total."""
    all_complaints: List[Dict[str, Any]] = []
    seen_odi: Set[str] = set()

    for fpath in COMPLAINT_FLAT_FILES:
        if not fpath.exists():
            continue
        print(f"    Reading {fpath.name}...")
        with open(fpath, "r") as f:
            data = json.load(f)
        complaints = data.get("complaints", [])
        for c in complaints:
            odi = str(c.get("odi_number", "")).strip()
            if not odi or odi in seen_odi:
                continue
            seen_odi.add(odi)
            all_complaints.append(c)
            if len(all_complaints) >= max_count:
                return all_complaints

    return all_complaints


def sync_complaints(conn: psycopg2.extensions.connection) -> int:
    """Load NHTSA complaints into vehicle_complaints table. Returns count."""
    print("\n=== NHTSA Complaints ===")

    # Try sampled file first
    if SAMPLED_COMPLAINTS_FILE.exists():
        print(f"  Loading from {SAMPLED_COMPLAINTS_FILE.name}...")
        with open(SAMPLED_COMPLAINTS_FILE, "r") as f:
            raw = json.load(f)
        if isinstance(raw, list):
            complaints = raw
        else:
            complaints = raw.get("complaints", raw.get("data", []))
    else:
        print("  sampled_200k.json not found, loading from flat files...")
        complaints = _load_complaints_from_flat_files(MAX_COMPLAINTS)

    total = len(complaints)
    print(f"  Total complaints to sync: {total}")

    sql = """
        INSERT INTO vehicle_complaints (
            odi_number, manufacturer, make, model, model_year,
            crash, fire, injuries, deaths,
            complaint_date, date_of_incident,
            components, summary, extracted_dtc_codes
        )
        VALUES %s
        ON CONFLICT (odi_number) DO UPDATE SET
            manufacturer = EXCLUDED.manufacturer,
            make = EXCLUDED.make,
            model = EXCLUDED.model,
            model_year = EXCLUDED.model_year,
            crash = EXCLUDED.crash,
            fire = EXCLUDED.fire,
            injuries = EXCLUDED.injuries,
            deaths = EXCLUDED.deaths,
            complaint_date = COALESCE(EXCLUDED.complaint_date, vehicle_complaints.complaint_date),
            date_of_incident = COALESCE(EXCLUDED.date_of_incident, vehicle_complaints.date_of_incident),
            components = COALESCE(EXCLUDED.components, vehicle_complaints.components),
            summary = COALESCE(EXCLUDED.summary, vehicle_complaints.summary),
            extracted_dtc_codes = EXCLUDED.extracted_dtc_codes,
            updated_at = NOW()
    """

    upserted = 0
    skipped = 0
    dtc_extracted_total = 0
    cur = conn.cursor()

    for i in tqdm(range(0, total, BATCH_COMPLAINTS), desc="  Complaints", unit="batch"):
        batch = complaints[i : i + BATCH_COMPLAINTS]
        values = []
        for c in batch:
            odi_number = str(c.get("odi_number", "")).strip()
            if not odi_number:
                skipped += 1
                continue

            manufacturer = str(c.get("manufacturer", "")).strip()[:100]
            make = str(c.get("make", "")).strip()[:50]
            model = str(c.get("model", "")).strip()[:100]

            # model_year: ensure it's an integer
            try:
                model_year = int(c.get("model_year", 0))
            except (ValueError, TypeError):
                skipped += 1
                continue
            if model_year < 1900 or model_year > 2030:
                skipped += 1
                continue

            if not manufacturer or not make or not model:
                skipped += 1
                continue

            crash = bool(c.get("crash", False))
            fire = bool(c.get("fire", False))
            try:
                injuries = int(c.get("injuries", 0) or 0)
            except (ValueError, TypeError):
                injuries = 0
            try:
                deaths = int(c.get("deaths", 0) or 0)
            except (ValueError, TypeError):
                deaths = 0

            # Dates
            complaint_date = parse_date_str(
                c.get("date_received") or c.get("complaint_date")
            )
            date_of_incident = parse_date_str(c.get("date_of_incident"))

            # Component
            components = c.get("component") or c.get("components")
            if components:
                components = str(components).strip()[:500]
            else:
                components = None

            # Summary + DTC extraction
            summary = c.get("summary")
            if summary:
                summary = str(summary).strip()
            dtc_codes = extract_dtc_codes(summary)
            if dtc_codes:
                dtc_extracted_total += len(dtc_codes)

            values.append(
                (
                    odi_number[:20],
                    manufacturer,
                    make,
                    model,
                    model_year,
                    crash,
                    fire,
                    injuries,
                    deaths,
                    complaint_date,
                    date_of_incident,
                    components,
                    summary,
                    dtc_codes,
                )
            )

        if values:
            psycopg2.extras.execute_values(cur, sql, values, page_size=BATCH_COMPLAINTS)
            upserted += len(values)

    conn.commit()
    cur.close()
    print(f"  Done: {upserted} complaints upserted, {skipped} skipped")
    print(f"  DTC codes extracted from summaries: {dtc_extracted_total}")
    return upserted


# ---------------------------------------------------------------------------
# EPA Vehicles sync
# ---------------------------------------------------------------------------
def sync_epa_vehicles(conn: psycopg2.extensions.connection) -> int:
    """Load EPA vehicle data into epa_vehicles table. Returns count."""
    print("\n=== EPA Vehicles ===")
    if not EPA_FILE.exists():
        print(f"  SKIP: {EPA_FILE} not found")
        return 0

    with open(EPA_FILE, "r") as f:
        vehicles = json.load(f)

    if isinstance(vehicles, dict):
        vehicles = vehicles.get("vehicles", vehicles.get("data", []))

    total = len(vehicles)
    print(f"  Source: {EPA_FILE.name} ({total} vehicles)")

    sql = """
        INSERT INTO epa_vehicles (
            id, make, model, base_model, model_year,
            vehicle_class, drive_type, transmission,
            cylinders, displacement_liters, engine_description,
            fuel_type, fuel_category,
            mpg_city, mpg_highway, mpg_combined,
            co2_grams_per_mile, ev_motor, range_miles,
            has_turbo, has_supercharger
        )
        VALUES %s
        ON CONFLICT (id) DO UPDATE SET
            make = EXCLUDED.make,
            model = EXCLUDED.model,
            base_model = EXCLUDED.base_model,
            model_year = EXCLUDED.model_year,
            vehicle_class = EXCLUDED.vehicle_class,
            drive_type = EXCLUDED.drive_type,
            transmission = EXCLUDED.transmission,
            cylinders = EXCLUDED.cylinders,
            displacement_liters = EXCLUDED.displacement_liters,
            engine_description = EXCLUDED.engine_description,
            fuel_type = EXCLUDED.fuel_type,
            fuel_category = EXCLUDED.fuel_category,
            mpg_city = EXCLUDED.mpg_city,
            mpg_highway = EXCLUDED.mpg_highway,
            mpg_combined = EXCLUDED.mpg_combined,
            co2_grams_per_mile = EXCLUDED.co2_grams_per_mile,
            ev_motor = EXCLUDED.ev_motor,
            range_miles = EXCLUDED.range_miles,
            has_turbo = EXCLUDED.has_turbo,
            has_supercharger = EXCLUDED.has_supercharger
    """

    upserted = 0
    cur = conn.cursor()

    for i in tqdm(range(0, total, BATCH_EPA), desc="  EPA vehicles", unit="batch"):
        batch = vehicles[i : i + BATCH_EPA]
        values = []
        for v in batch:
            epa_id = v.get("id")
            if epa_id is None:
                continue
            try:
                epa_id = int(epa_id)
            except (ValueError, TypeError):
                continue

            make = str(v.get("make", "")).strip()[:100]
            model = str(v.get("model", "")).strip()[:200]
            base_model = v.get("base_model")
            if base_model:
                base_model = str(base_model).strip()[:200]

            try:
                model_year = int(v.get("model_year", 0))
            except (ValueError, TypeError):
                continue
            if model_year < 1900 or model_year > 2030:
                continue

            if not make or not model:
                continue

            vehicle_class = v.get("vehicle_class")
            if vehicle_class:
                vehicle_class = str(vehicle_class)[:100]
            drive_type = v.get("drive_type")
            if drive_type:
                drive_type = str(drive_type)[:100]
            transmission = v.get("transmission")
            if transmission:
                transmission = str(transmission)[:100]

            cylinders = v.get("cylinders")
            if cylinders is not None:
                try:
                    cylinders = int(cylinders)
                except (ValueError, TypeError):
                    cylinders = None

            displacement = v.get("displacement_liters")
            if displacement is not None:
                try:
                    displacement = float(displacement)
                except (ValueError, TypeError):
                    displacement = None

            engine_desc = v.get("engine_description")
            if engine_desc:
                engine_desc = str(engine_desc)[:300]

            fuel_type = v.get("fuel_type")
            if fuel_type:
                fuel_type = str(fuel_type)[:50]
            fuel_category = v.get("fuel_category")
            if fuel_category:
                fuel_category = str(fuel_category)[:100]

            mpg_city = _safe_int(v.get("mpg_city"))
            mpg_highway = _safe_int(v.get("mpg_highway"))
            mpg_combined = _safe_int(v.get("mpg_combined"))

            co2 = v.get("co2_grams_per_mile")
            if co2 is not None:
                try:
                    co2 = float(co2)
                except (ValueError, TypeError):
                    co2 = None

            ev_motor = v.get("ev_motor")
            if ev_motor:
                ev_motor = str(ev_motor).strip()[:200]
                if not ev_motor:
                    ev_motor = None

            range_miles = _safe_int(v.get("range_miles"))
            if range_miles == 0:
                range_miles = None

            has_turbo = bool(v.get("has_turbo", False))
            has_supercharger = bool(v.get("has_supercharger", False))

            values.append(
                (
                    epa_id,
                    make,
                    model,
                    base_model,
                    model_year,
                    vehicle_class,
                    drive_type,
                    transmission,
                    cylinders,
                    displacement,
                    engine_desc,
                    fuel_type,
                    fuel_category,
                    mpg_city,
                    mpg_highway,
                    mpg_combined,
                    co2,
                    ev_motor,
                    range_miles,
                    has_turbo,
                    has_supercharger,
                )
            )

        if values:
            psycopg2.extras.execute_values(cur, sql, values, page_size=BATCH_EPA)
            upserted += len(values)

    conn.commit()
    cur.close()
    print(f"  Done: {upserted} EPA vehicles upserted")
    return upserted


def _safe_int(val: Any) -> Optional[int]:
    """Convert value to int, or return None."""
    if val is None:
        return None
    try:
        result = int(val)
        return result if result > 0 else None
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Sprint 9 PostgreSQL Data Sync")
    parser.add_argument("--all", action="store_true", help="Sync everything")
    parser.add_argument("--dtc", action="store_true", help="Sync DTC codes only")
    parser.add_argument(
        "--makes", action="store_true", help="Sync vehicle makes+models"
    )
    parser.add_argument("--complaints", action="store_true", help="Sync complaints")
    parser.add_argument("--epa", action="store_true", help="Sync EPA vehicles")
    parser.add_argument("--reset", action="store_true", help="Reset checkpoint")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore checkpoint, re-sync even if marked done",
    )
    args = parser.parse_args()

    if args.reset:
        reset_checkpoint()
        return

    # Default to --all if nothing specified
    if not any([args.all, args.dtc, args.makes, args.complaints, args.epa]):
        args.all = True

    ckpt = load_checkpoint()
    start_time = time.time()

    print("=" * 60)
    print("  Sprint 9 PostgreSQL Data Sync")
    print("=" * 60)

    conn = get_connection()
    stats: Dict[str, int] = {}

    try:
        # DTC Codes
        if args.all or args.dtc:
            if ckpt.get("dtc_done") and not args.force:
                print("\n  DTC codes: already done (use --force to re-sync)")
            else:
                count = sync_dtc_codes(conn)
                stats["dtc_codes"] = count
                ckpt["dtc_done"] = True
                save_checkpoint(ckpt)

        # Vehicle Makes + Models (makes must come first due to FK)
        if args.all or args.makes:
            if ckpt.get("makes_done") and ckpt.get("models_done") and not args.force:
                print("\n  Makes+Models: already done (use --force to re-sync)")
            else:
                count_makes, makes_data = sync_vehicle_makes(conn)
                stats["vehicle_makes"] = count_makes
                ckpt["makes_done"] = True
                save_checkpoint(ckpt)

                count_models = sync_vehicle_models(conn, makes_data)
                stats["vehicle_models"] = count_models
                ckpt["models_done"] = True
                save_checkpoint(ckpt)

        # Complaints
        if args.all or args.complaints:
            if ckpt.get("complaints_done") and not args.force:
                print("\n  Complaints: already done (use --force to re-sync)")
            else:
                count = sync_complaints(conn)
                stats["complaints"] = count
                ckpt["complaints_done"] = True
                ckpt["complaints_loaded"] = count
                save_checkpoint(ckpt)

        # EPA Vehicles
        if args.all or args.epa:
            if ckpt.get("epa_done") and not args.force:
                print("\n  EPA vehicles: already done (use --force to re-sync)")
            else:
                count = sync_epa_vehicles(conn)
                stats["epa_vehicles"] = count
                ckpt["epa_done"] = True
                save_checkpoint(ckpt)

    except KeyboardInterrupt:
        print("\n\n  Interrupted! Saving checkpoint...")
        save_checkpoint(ckpt)
        conn.close()
        sys.exit(1)
    except Exception as exc:
        print(f"\n  ERROR: {exc}")
        save_checkpoint(ckpt)
        conn.close()
        raise
    finally:
        if not conn.closed:
            conn.close()

    elapsed = time.time() - start_time

    # Summary
    print("\n" + "=" * 60)
    print("  SYNC COMPLETE")
    print("=" * 60)
    for table, count in stats.items():
        print(f"  {table:25s}: {count:>10,}")
    print(f"  {'elapsed':25s}: {elapsed:>10.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
