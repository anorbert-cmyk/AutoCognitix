#!/usr/bin/env python3
"""
NHTSA Vehicle Makes & Models Complete Sync Script for AutoCognitix.

Downloads ALL vehicle makes from NHTSA vPIC API and models for the
top 100 most important makes (European, Asian, American markets).

Features:
- Downloads all makes (~10,000+)
- Downloads models for top 100 prioritized makes
- Checkpoint/resume support (survives interruptions)
- Rate limiting (0.3s between requests)
- Timeout handling (30s per request)
- Merges with Back4App and Wikipedia data sources
- Produces normalized master vehicle file

Output files:
- data/nhtsa/all_makes_complete.json       - All NHTSA makes
- data/nhtsa/all_models_complete.json       - Top 100 makes' models
- data/nhtsa/models_by_make/<Make>.json     - Per-make model files
- data/vehicles/vehicles_master.json        - Merged master file

Usage:
    python scripts/sync_nhtsa_vehicles_complete.py
    python scripts/sync_nhtsa_vehicles_complete.py --resume
    python scripts/sync_nhtsa_vehicles_complete.py --makes-only
    python scripts/sync_nhtsa_vehicles_complete.py --models-only
    python scripts/sync_nhtsa_vehicles_complete.py --merge-only
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import httpx

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# =============================================================================
# Configuration
# =============================================================================

VPIC_BASE_URL = "https://vpic.nhtsa.dot.gov/api/vehicles"
GET_ALL_MAKES_URL = f"{VPIC_BASE_URL}/GetAllMakes"
GET_MODELS_FOR_MAKE_URL = f"{VPIC_BASE_URL}/GetModelsForMake"

# Rate limiting: 0.3s between requests (conservative for NHTSA)
REQUEST_DELAY = 0.3
REQUEST_TIMEOUT = 30.0
MAX_RETRIES = 3
RETRY_DELAY = 5.0

# Output paths
DATA_DIR = PROJECT_ROOT / "data"
NHTSA_DIR = DATA_DIR / "nhtsa"
VEHICLES_DIR = DATA_DIR / "vehicles"
MODELS_BY_MAKE_DIR = NHTSA_DIR / "models_by_make"
CHECKPOINT_DIR = PROJECT_ROOT / "scripts" / "checkpoints"

MAKES_OUTPUT = NHTSA_DIR / "all_makes_complete.json"
MODELS_OUTPUT = NHTSA_DIR / "all_models_complete.json"
MASTER_OUTPUT = VEHICLES_DIR / "vehicles_master.json"
CHECKPOINT_FILE = CHECKPOINT_DIR / "vehicle_sync.json"
LOG_FILE = NHTSA_DIR / "sync_complete.log"

# Existing data sources for merge
BACK4APP_FILE = VEHICLES_DIR / "back4app_vehicles.json"
WIKIPEDIA_FILE = VEHICLES_DIR / "wikipedia_vehicles.json"

# Top 100 makes - prioritized for AutoCognitix (European focus)
TOP_100_MAKES = [
    # EU - Primary (most common in Hungary/Europe)
    "BMW", "Mercedes-Benz", "Audi", "Volkswagen", "Opel", "Skoda",
    "Seat", "Renault", "Peugeot", "Citroen", "Fiat", "Volvo",
    "Saab", "Alfa Romeo", "Lancia", "Porsche", "Mini", "Smart",
    "Dacia", "DS",
    # EU - Additional
    "Vauxhall", "Rover", "MG", "Cupra", "Alpine",
    # JP
    "Toyota", "Honda", "Nissan", "Mazda", "Subaru", "Mitsubishi",
    "Suzuki", "Lexus", "Infiniti", "Acura", "Isuzu", "Daihatsu",
    # KR
    "Hyundai", "Kia", "Genesis", "SsangYong",
    # US
    "Ford", "Chevrolet", "GMC", "Dodge", "Chrysler", "Jeep", "Ram",
    "Cadillac", "Buick", "Lincoln", "Tesla", "Pontiac", "Saturn",
    "Oldsmobile", "Mercury", "Hummer",
    # Premium / Exotic
    "Land Rover", "Jaguar", "Bentley", "Rolls-Royce", "Maserati",
    "Ferrari", "Lamborghini", "Aston Martin", "McLaren", "Lotus",
    "Bugatti", "Pagani",
    # Chinese (growing EU presence)
    "BYD", "MG", "NIO", "Polestar", "Lynk & Co",
    # Other notable
    "Rivian", "Lucid", "Fisker", "Lada", "Tata",
    "Mahindra", "Proton", "Perodua", "Great Wall",
    "Chery", "Geely", "BAIC", "Haval",
    # Trucks / Commercial
    "Freightliner", "Kenworth", "Peterbilt", "International",
    "Mack", "Western Star", "Volvo Trucks",
    # Motorcycles (popular in diagnostics)
    "Harley-Davidson", "Indian",
]

# Deduplicate while preserving order
_seen: Set[str] = set()
TOP_100_MAKES_UNIQUE: List[str] = []
for _m in TOP_100_MAKES:
    if _m.lower() not in _seen:
        _seen.add(_m.lower())
        TOP_100_MAKES_UNIQUE.append(_m)
TOP_100_MAKES = TOP_100_MAKES_UNIQUE


# =============================================================================
# Logging
# =============================================================================

def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging."""
    log = logging.getLogger("nhtsa_complete_sync")
    log.setLevel(logging.DEBUG if verbose else logging.INFO)
    log.handlers.clear()

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    console.setFormatter(fmt)
    log.addHandler(console)

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(fh)

    return log


logger = setup_logging()


# =============================================================================
# Checkpoint Management
# =============================================================================

def load_checkpoint() -> Dict[str, Any]:
    """Load checkpoint from file."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "makes_downloaded": False,
        "models_completed_makes": [],
        "models_failed_makes": [],
        "last_updated": None,
    }


def save_checkpoint(checkpoint: Dict[str, Any]) -> None:
    """Save checkpoint to file."""
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    checkpoint["last_updated"] = datetime.now(timezone.utc).isoformat()
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)


def save_json(data: Any, path: Path) -> None:
    """Save data to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Optional[Any]:
    """Load JSON from file."""
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# NHTSA API Client (synchronous httpx for simplicity)
# =============================================================================

class NHTSAClient:
    """Synchronous NHTSA vPIC API client with rate limiting and retries."""

    def __init__(self):
        self._client = httpx.Client(
            timeout=httpx.Timeout(REQUEST_TIMEOUT),
            headers={
                "User-Agent": "AutoCognitix/2.0 (Vehicle Diagnostic Platform)",
                "Accept": "application/json",
            },
        )
        self._last_request_time = 0.0
        self.total_requests = 0
        self.total_errors = 0

    def _rate_limit(self) -> None:
        """Enforce rate limit between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()

    def get(self, url: str, params: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
        """Make a GET request with rate limiting and retries."""
        for attempt in range(MAX_RETRIES):
            self._rate_limit()
            self.total_requests += 1

            try:
                response = self._client.get(url, params=params)

                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 60))
                    logger.warning(f"Rate limited! Waiting {retry_after}s...")
                    time.sleep(retry_after)
                    continue

                if response.status_code >= 500:
                    logger.warning(
                        f"Server error {response.status_code} for {url}, "
                        f"retry {attempt + 1}/{MAX_RETRIES}"
                    )
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue

                response.raise_for_status()
                return response.json()

            except httpx.TimeoutException:
                logger.warning(f"Timeout for {url}, retry {attempt + 1}/{MAX_RETRIES}")
                self.total_errors += 1
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                continue

            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP {e.response.status_code} for {url}")
                self.total_errors += 1
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                continue

            except Exception as e:
                logger.error(f"Request error for {url}: {e}")
                self.total_errors += 1
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                continue

        return None

    def close(self) -> None:
        """Close client."""
        self._client.close()


# =============================================================================
# Core Sync Functions
# =============================================================================

def download_all_makes(client: NHTSAClient) -> List[Dict[str, Any]]:
    """Download all makes from NHTSA vPIC API."""
    logger.info("=" * 60)
    logger.info("Downloading ALL vehicle makes from NHTSA...")
    logger.info("=" * 60)

    data = client.get(GET_ALL_MAKES_URL, {"format": "json"})
    if not data:
        logger.error("Failed to fetch makes from NHTSA API")
        return []

    results = data.get("Results", [])
    makes = []

    for item in results:
        make_id = item.get("Make_ID")
        make_name = item.get("Make_Name")
        if make_id is not None and make_name:
            makes.append({
                "make_id": int(make_id),
                "make_name": make_name.strip(),
            })

    logger.info(f"Downloaded {len(makes)} vehicle makes")

    # Save
    output = {
        "metadata": {
            "source": "NHTSA vPIC API",
            "api_url": GET_ALL_MAKES_URL,
            "synced_at": datetime.now(timezone.utc).isoformat(),
            "total_makes": len(makes),
        },
        "makes": makes,
    }
    save_json(output, MAKES_OUTPUT)
    logger.info(f"Saved to {MAKES_OUTPUT}")

    return makes


def download_models_for_make(
    client: NHTSAClient, make_name: str
) -> List[Dict[str, Any]]:
    """Download all models for a specific make."""
    url = f"{GET_MODELS_FOR_MAKE_URL}/{make_name}"
    data = client.get(url, {"format": "json"})

    if not data:
        return []

    results = data.get("Results", [])
    models = []

    for item in results:
        model_id = item.get("Model_ID")
        model_name = item.get("Model_Name")
        make_id = item.get("Make_ID")

        if model_id is not None and model_name:
            models.append({
                "model_id": int(model_id),
                "model_name": model_name.strip(),
                "make_id": int(make_id) if make_id else None,
                "make_name": item.get("Make_Name", make_name).strip(),
            })

    return models


def download_all_models(
    client: NHTSAClient,
    checkpoint: Dict[str, Any],
    all_nhtsa_makes: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Download models for top 100 makes with checkpoint support."""
    logger.info("=" * 60)
    logger.info(f"Downloading models for top {len(TOP_100_MAKES)} makes...")
    logger.info("=" * 60)

    # Build a lookup from NHTSA makes for exact name matching
    nhtsa_make_names = {m["make_name"].lower(): m["make_name"] for m in all_nhtsa_makes}

    completed = set(checkpoint.get("models_completed_makes", []))
    failed = checkpoint.get("models_failed_makes", [])
    all_models: Dict[str, List[Dict[str, Any]]] = {}

    # Load already-downloaded per-make files
    for make_name in completed:
        safe_name = make_name.replace("/", "_").replace("\\", "_")
        make_file = MODELS_BY_MAKE_DIR / f"{safe_name}.json"
        if make_file.exists():
            existing = load_json(make_file)
            if existing and "models" in existing:
                all_models[make_name] = existing["models"]

    # Determine which makes still need downloading
    remaining = [m for m in TOP_100_MAKES if m not in completed]

    if not remaining:
        logger.info("All target makes already completed (checkpoint)")
        total_models = sum(len(v) for v in all_models.values())
        logger.info(f"Total models from checkpoint: {total_models}")
        return all_models

    logger.info(f"Already completed: {len(completed)}, Remaining: {len(remaining)}")

    for i, make_name in enumerate(remaining, 1):
        # Find exact NHTSA name (case-insensitive match)
        exact_name = nhtsa_make_names.get(make_name.lower(), make_name)

        logger.info(f"[{i}/{len(remaining)}] Fetching models for: {exact_name}")

        try:
            models = download_models_for_make(client, exact_name)

            if models:
                all_models[exact_name] = models
                logger.info(f"  -> {len(models)} models found")

                # Save per-make file
                safe_name = exact_name.replace("/", "_").replace("\\", "_")
                make_file = MODELS_BY_MAKE_DIR / f"{safe_name}.json"
                save_json({
                    "make_name": exact_name,
                    "synced_at": datetime.now(timezone.utc).isoformat(),
                    "total_models": len(models),
                    "models": models,
                }, make_file)
            else:
                logger.warning(f"  -> No models found for {exact_name}")
                # Try the original name if it differs
                if exact_name != make_name:
                    logger.info(f"  -> Trying original name: {make_name}")
                    models = download_models_for_make(client, make_name)
                    if models:
                        all_models[make_name] = models
                        logger.info(f"  -> {len(models)} models found with original name")
                        safe_name = make_name.replace("/", "_").replace("\\", "_")
                        make_file = MODELS_BY_MAKE_DIR / f"{safe_name}.json"
                        save_json({
                            "make_name": make_name,
                            "synced_at": datetime.now(timezone.utc).isoformat(),
                            "total_models": len(models),
                            "models": models,
                        }, make_file)

            # Update checkpoint
            checkpoint["models_completed_makes"].append(make_name)
            save_checkpoint(checkpoint)

        except Exception as e:
            logger.error(f"  -> FAILED: {e}")
            failed.append(make_name)
            checkpoint["models_failed_makes"] = failed
            save_checkpoint(checkpoint)

    # Save combined models file
    total_models = sum(len(v) for v in all_models.values())
    output = {
        "metadata": {
            "source": "NHTSA vPIC API",
            "synced_at": datetime.now(timezone.utc).isoformat(),
            "total_makes": len(all_models),
            "total_models": total_models,
            "target_makes_count": len(TOP_100_MAKES),
        },
        "models_by_make": {
            make: models for make, models in sorted(all_models.items())
        },
    }
    save_json(output, MODELS_OUTPUT)
    logger.info(f"Saved {total_models} models across {len(all_models)} makes to {MODELS_OUTPUT}")

    if failed:
        logger.warning(f"Failed makes: {failed}")

    return all_models


# =============================================================================
# Merge Logic
# =============================================================================

def merge_data_sources(
    nhtsa_makes: List[Dict[str, Any]],
    nhtsa_models: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Merge NHTSA, Back4App, and Wikipedia data into a master file."""
    logger.info("=" * 60)
    logger.info("Merging data sources into vehicles_master.json...")
    logger.info("=" * 60)

    # Master structure: keyed by lowercase make name
    master: Dict[str, Dict[str, Any]] = {}

    # --- 1. NHTSA Makes ---
    nhtsa_make_count = 0
    for make in nhtsa_makes:
        key = make["make_name"].lower().strip()
        if key not in master:
            master[key] = {
                "make_name": make["make_name"].strip(),
                "make_id_nhtsa": make["make_id"],
                "sources": ["nhtsa"],
                "models": {},
            }
            nhtsa_make_count += 1
        else:
            if "nhtsa" not in master[key]["sources"]:
                master[key]["sources"].append("nhtsa")
            master[key]["make_id_nhtsa"] = make["make_id"]

    logger.info(f"NHTSA makes added: {nhtsa_make_count}")

    # --- 2. NHTSA Models ---
    nhtsa_model_count = 0
    for make_name, models in nhtsa_models.items():
        key = make_name.lower().strip()
        if key not in master:
            master[key] = {
                "make_name": make_name.strip(),
                "make_id_nhtsa": None,
                "sources": ["nhtsa"],
                "models": {},
            }
        entry = master[key]

        for model in models:
            model_key = model["model_name"].lower().strip()
            if model_key not in entry["models"]:
                entry["models"][model_key] = {
                    "model_name": model["model_name"].strip(),
                    "model_id_nhtsa": model["model_id"],
                    "sources": ["nhtsa"],
                    "years": [],
                }
                nhtsa_model_count += 1
            else:
                existing = entry["models"][model_key]
                existing["model_id_nhtsa"] = model["model_id"]
                if "nhtsa" not in existing["sources"]:
                    existing["sources"].append("nhtsa")

            if model.get("make_id") and not entry.get("make_id_nhtsa"):
                entry["make_id_nhtsa"] = model["make_id"]

    logger.info(f"NHTSA models added: {nhtsa_model_count}")

    # --- 3. Back4App ---
    back4app_model_count = 0
    back4app = load_json(BACK4APP_FILE)
    if back4app and "makes" in back4app:
        for make_data in back4app["makes"]:
            make_name = make_data.get("name", "").strip()
            if not make_name:
                continue
            key = make_name.lower()

            if key not in master:
                master[key] = {
                    "make_name": make_name,
                    "make_id_nhtsa": None,
                    "sources": ["back4app"],
                    "models": {},
                }
            else:
                if "back4app" not in master[key]["sources"]:
                    master[key]["sources"].append("back4app")

            entry = master[key]

            for model_data in make_data.get("models", []):
                model_name = model_data.get("name", "").strip()
                if not model_name:
                    continue
                model_key = model_name.lower()
                years = model_data.get("years", [])

                if model_key not in entry["models"]:
                    entry["models"][model_key] = {
                        "model_name": model_name,
                        "model_id_nhtsa": None,
                        "sources": ["back4app"],
                        "years": sorted(set(years)),
                    }
                    back4app_model_count += 1
                else:
                    existing = entry["models"][model_key]
                    if "back4app" not in existing["sources"]:
                        existing["sources"].append("back4app")
                    # Merge years
                    merged_years = sorted(set(existing["years"] + years))
                    existing["years"] = merged_years

        logger.info(f"Back4App models added: {back4app_model_count}")
    else:
        logger.warning("Back4App data not found or empty")

    # --- 4. Wikipedia ---
    wikipedia_model_count = 0
    wikipedia = load_json(WIKIPEDIA_FILE)
    if wikipedia and "vehicles" in wikipedia:
        for vehicle in wikipedia["vehicles"]:
            make_name = vehicle.get("make", "").strip()
            model_name = vehicle.get("model", "").strip()
            if not make_name or not model_name:
                continue

            key = make_name.lower()

            if key not in master:
                master[key] = {
                    "make_name": make_name,
                    "make_id_nhtsa": None,
                    "sources": ["wikipedia"],
                    "models": {},
                }
            else:
                if "wikipedia" not in master[key]["sources"]:
                    master[key]["sources"].append("wikipedia")

            entry = master[key]
            model_key = model_name.lower()

            years = []
            ys = vehicle.get("years_start")
            ye = vehicle.get("years_end")
            if ys:
                years.append(int(ys))
            if ye:
                years.append(int(ye))

            if model_key not in entry["models"]:
                entry["models"][model_key] = {
                    "model_name": model_name,
                    "model_id_nhtsa": None,
                    "sources": ["wikipedia"],
                    "years": sorted(set(years)),
                    "wikipedia_url": vehicle.get("wikipedia_url"),
                }
                wikipedia_model_count += 1
            else:
                existing = entry["models"][model_key]
                if "wikipedia" not in existing["sources"]:
                    existing["sources"].append("wikipedia")
                if vehicle.get("wikipedia_url"):
                    existing["wikipedia_url"] = vehicle["wikipedia_url"]
                if years:
                    merged_years = sorted(set(existing.get("years", []) + years))
                    existing["years"] = merged_years

        logger.info(f"Wikipedia models added: {wikipedia_model_count}")
    else:
        logger.warning("Wikipedia data not found or empty")

    # --- 5. Convert to output format ---
    makes_list = []
    total_models = 0
    for key in sorted(master.keys()):
        entry = master[key]
        models_list = sorted(
            entry["models"].values(),
            key=lambda m: m["model_name"].lower(),
        )
        total_models += len(models_list)

        makes_list.append({
            "make_name": entry["make_name"],
            "make_id_nhtsa": entry.get("make_id_nhtsa"),
            "sources": entry["sources"],
            "model_count": len(models_list),
            "models": models_list,
        })

    output = {
        "metadata": {
            "description": "AutoCognitix Master Vehicle Database",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_makes": len(makes_list),
            "total_models": total_models,
            "data_sources": {
                "nhtsa": {
                    "makes": nhtsa_make_count,
                    "models": nhtsa_model_count,
                },
                "back4app": {
                    "models": back4app_model_count,
                },
                "wikipedia": {
                    "models": wikipedia_model_count,
                },
            },
        },
        "makes": makes_list,
    }

    save_json(output, MASTER_OUTPUT)
    logger.info(f"Master file saved: {MASTER_OUTPUT}")
    logger.info(f"Total makes: {len(makes_list)}, Total models: {total_models}")

    return output


# =============================================================================
# Statistics
# =============================================================================

def print_statistics(
    nhtsa_makes: List[Dict[str, Any]],
    nhtsa_models: Dict[str, List[Dict[str, Any]]],
    master: Dict[str, Any],
    client: Optional[NHTSAClient],
    elapsed: float,
) -> None:
    """Print comprehensive statistics."""
    print("\n" + "=" * 70)
    print("  NHTSA VEHICLE SYNC - COMPLETE STATISTICS")
    print("=" * 70)

    # Before
    old_makes = load_json(NHTSA_DIR / "all_makes.json")
    old_models = load_json(NHTSA_DIR / "all_models.json")
    old_makes_count = old_makes.get("metadata", {}).get("total_makes", 0) if old_makes else 0
    old_models_count = old_models.get("metadata", {}).get("total_models", 0) if old_models else 0

    # After
    new_makes_count = len(nhtsa_makes)
    new_models_count = sum(len(v) for v in nhtsa_models.values())
    master_meta = master.get("metadata", {}) if master else {}

    print("\n  BEFORE (previous sync):")
    print(f"    Makes:  {old_makes_count}")
    print(f"    Models: {old_models_count} (only Toyota)")

    print("\n  AFTER (this sync):")
    print(f"    NHTSA Makes:  {new_makes_count}")
    print(f"    NHTSA Models: {new_models_count} (top {len(TOP_100_MAKES)} makes)")

    print("\n  IMPROVEMENT:")
    if old_makes_count > 0:
        print(f"    Makes:  {old_makes_count} -> {new_makes_count} ({new_makes_count / old_makes_count:.0f}x increase)")
    else:
        print(f"    Makes:  0 -> {new_makes_count}")
    if old_models_count > 0:
        print(f"    Models: {old_models_count} -> {new_models_count} ({new_models_count / old_models_count:.0f}x increase)")
    else:
        print(f"    Models: 0 -> {new_models_count}")

    print("\n  MASTER FILE (merged):")
    print(f"    Total Makes:  {master_meta.get('total_makes', 'N/A')}")
    print(f"    Total Models: {master_meta.get('total_models', 'N/A')}")
    ds = master_meta.get("data_sources", {})
    print(f"    NHTSA:     {ds.get('nhtsa', {}).get('makes', 0)} makes, {ds.get('nhtsa', {}).get('models', 0)} models")
    print(f"    Back4App:  {ds.get('back4app', {}).get('models', 0)} models")
    print(f"    Wikipedia: {ds.get('wikipedia', {}).get('models', 0)} models")

    # Top makes with most models
    if nhtsa_models:
        print("\n  TOP 20 MAKES BY MODEL COUNT:")
        sorted_makes = sorted(nhtsa_models.items(), key=lambda x: len(x[1]), reverse=True)
        for rank, (make, models) in enumerate(sorted_makes[:20], 1):
            print(f"    {rank:2d}. {make:<25s} {len(models):>4d} models")

    if client:
        print("\n  API STATISTICS:")
        print(f"    Total requests: {client.total_requests}")
        print(f"    Total errors:   {client.total_errors}")
        print(f"    Elapsed time:   {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    print("\n  OUTPUT FILES:")
    for path in [MAKES_OUTPUT, MODELS_OUTPUT, MASTER_OUTPUT]:
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"    {path.name:<35s} {size_kb:>8.1f} KB")
    if MODELS_BY_MAKE_DIR.exists():
        count = len(list(MODELS_BY_MAKE_DIR.glob("*.json")))
        print(f"    models_by_make/                    {count} files")

    print("=" * 70)


# =============================================================================
# Main
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download ALL NHTSA vehicle makes and models for top 100 makes"
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--makes-only", action="store_true", help="Only download makes")
    parser.add_argument("--models-only", action="store_true", help="Only download models (assumes makes exist)")
    parser.add_argument("--merge-only", action="store_true", help="Only merge existing data")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    global logger
    logger = setup_logging(verbose=args.verbose)

    start_time = time.time()

    # Ensure directories exist
    for d in [NHTSA_DIR, VEHICLES_DIR, MODELS_BY_MAKE_DIR, CHECKPOINT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    checkpoint = load_checkpoint() if args.resume else {
        "makes_downloaded": False,
        "models_completed_makes": [],
        "models_failed_makes": [],
        "last_updated": None,
    }

    if args.resume:
        logger.info(f"Resuming from checkpoint: {len(checkpoint.get('models_completed_makes', []))} makes completed")

    client = None
    nhtsa_makes = []
    nhtsa_models: Dict[str, List[Dict[str, Any]]] = {}
    master = {}

    try:
        # --- Merge Only Mode ---
        if args.merge_only:
            # Load existing data
            makes_data = load_json(MAKES_OUTPUT)
            if makes_data:
                nhtsa_makes = makes_data.get("makes", [])
            models_data = load_json(MODELS_OUTPUT)
            if models_data:
                nhtsa_models = models_data.get("models_by_make", {})

            master = merge_data_sources(nhtsa_makes, nhtsa_models)
            elapsed = time.time() - start_time
            print_statistics(nhtsa_makes, nhtsa_models, master, None, elapsed)
            return

        client = NHTSAClient()

        # --- Step 1: Download Makes ---
        if not args.models_only:
            if args.resume and checkpoint.get("makes_downloaded"):
                logger.info("Makes already downloaded (checkpoint), loading from file...")
                makes_data = load_json(MAKES_OUTPUT)
                if makes_data:
                    nhtsa_makes = makes_data.get("makes", [])
                    logger.info(f"Loaded {len(nhtsa_makes)} makes from file")
                else:
                    nhtsa_makes = download_all_makes(client)
            else:
                nhtsa_makes = download_all_makes(client)

            checkpoint["makes_downloaded"] = True
            save_checkpoint(checkpoint)

            if args.makes_only:
                elapsed = time.time() - start_time
                master = merge_data_sources(nhtsa_makes, nhtsa_models)
                print_statistics(nhtsa_makes, nhtsa_models, master, client, elapsed)
                return
        else:
            # Load existing makes for models-only mode
            makes_data = load_json(MAKES_OUTPUT)
            if makes_data:
                nhtsa_makes = makes_data.get("makes", [])
                logger.info(f"Loaded {len(nhtsa_makes)} existing makes")
            else:
                logger.error("No makes file found. Run without --models-only first.")
                sys.exit(1)

        # --- Step 2: Download Models ---
        if not args.makes_only:
            nhtsa_models = download_all_models(client, checkpoint, nhtsa_makes)

        # --- Step 3: Merge ---
        master = merge_data_sources(nhtsa_makes, nhtsa_models)

        elapsed = time.time() - start_time
        print_statistics(nhtsa_makes, nhtsa_models, master, client, elapsed)

    except KeyboardInterrupt:
        logger.warning("\nInterrupted! Progress saved to checkpoint.")
        save_checkpoint(checkpoint)
        sys.exit(1)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        save_checkpoint(checkpoint)
        raise

    finally:
        if client:
            client.close()


if __name__ == "__main__":
    main()
