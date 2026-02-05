#!/usr/bin/env python3
"""
OBDb GitHub Organization Data Importer.

This script imports vehicle-specific DTC codes and signal data from the OBDb
GitHub organization (https://github.com/OBDb), which contains 742+ vehicle
repositories with detailed diagnostic information.

Source:
    - Organization: https://github.com/OBDb
    - Format: JSON (signalsets/v3/default.json)
    - License: CC BY-SA 4.0

Usage:
    python scripts/import_obdb.py --list              # List available repos
    python scripts/import_obdb.py --download          # Download data
    python scripts/import_obdb.py --postgres          # Import to PostgreSQL
    python scripts/import_obdb.py --all               # Download and import all
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Data paths
DATA_DIR = PROJECT_ROOT / "data" / "vehicles"
CACHE_DIR = DATA_DIR / "obdb_cache"
REPOS_CACHE = CACHE_DIR / "repos.json"
VEHICLES_CACHE = CACHE_DIR / "vehicles.json"

# GitHub API configuration
GITHUB_API = "https://api.github.com"
GITHUB_RAW = "https://raw.githubusercontent.com"
OBDB_ORG = "OBDb"

# Rate limiting
RATE_LIMIT_DELAY = 0.5  # seconds between API requests
MAX_REPOS = 200  # Maximum repos to process (OBDb has 742+)

# Priority European makes to process first
PRIORITY_MAKES = [
    "Volkswagen", "VW", "Audi", "Skoda", "SEAT", "Porsche",
    "BMW", "Mercedes", "Mercedes-Benz",
    "Renault", "Peugeot", "Citroen",
    "Opel", "Vauxhall",
    "Fiat", "Alfa Romeo", "Lancia",
    "Volvo", "Saab",
    "Jaguar", "Land Rover", "Mini",
    "Ford", "Toyota", "Honda", "Nissan", "Mazda",  # Also popular in Europe
]


def get_github_token() -> Optional[str]:
    """Get GitHub token from environment."""
    return os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")


async def list_obdb_repos(client: httpx.AsyncClient) -> List[Dict[str, Any]]:
    """
    List all repositories in the OBDb organization.

    Args:
        client: HTTP client instance.

    Returns:
        List of repository information dictionaries.
    """
    logger.info("Fetching OBDb repository list...")

    repos = []
    page = 1
    per_page = 100

    headers = {"Accept": "application/vnd.github.v3+json"}
    token = get_github_token()
    if token:
        headers["Authorization"] = f"token {token}"
        logger.info("Using GitHub token for authentication")

    while True:
        url = f"{GITHUB_API}/orgs/{OBDB_ORG}/repos"
        params = {"page": page, "per_page": per_page, "type": "public"}

        try:
            response = await client.get(url, headers=headers, params=params)

            if response.status_code == 403:
                logger.warning("GitHub API rate limit reached")
                break

            if response.status_code != 200:
                logger.error(f"GitHub API error: {response.status_code}")
                break

            page_repos = response.json()

            if not page_repos:
                break

            repos.extend(page_repos)
            logger.info(f"Fetched page {page}: {len(page_repos)} repos")

            page += 1
            await asyncio.sleep(RATE_LIMIT_DELAY)

        except Exception as e:
            logger.error(f"Error fetching repos: {e}")
            break

    logger.info(f"Found {len(repos)} total repositories")
    return repos


def parse_repo_name(name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse vehicle make and model from repository name.

    OBDb repo names follow pattern: "Make-Model" or "Make_Model"
    Examples: "Volkswagen-Golf", "BMW-3-Series", "Mercedes-Benz-C-Class"

    Args:
        name: Repository name.

    Returns:
        Tuple of (make, model) or (None, None) if parsing fails.
    """
    # Common patterns
    patterns = [
        r'^([A-Za-z]+(?:-[A-Za-z]+)?)-(.+)$',  # Make-Model
        r'^([A-Za-z]+(?:_[A-Za-z]+)?)[_-](.+)$',  # Make_Model
    ]

    for pattern in patterns:
        match = re.match(pattern, name)
        if match:
            make = match.group(1).replace("-", " ").replace("_", " ")
            model = match.group(2).replace("-", " ").replace("_", " ")
            return make, model

    return None, None


async def download_signalset(
    client: httpx.AsyncClient,
    repo_name: str,
    owner: str = OBDB_ORG,
) -> Optional[Dict[str, Any]]:
    """
    Download the signalset JSON from a repository.

    Args:
        client: HTTP client instance.
        repo_name: Repository name.
        owner: Repository owner (default: OBDb).

    Returns:
        Signalset data dictionary or None if not found.
    """
    # Try different file paths
    paths = [
        "signalsets/v3/default.json",
        "signalsets/v2/default.json",
        "signalsets/default.json",
        "default.json",
    ]

    for path in paths:
        url = f"{GITHUB_RAW}/{owner}/{repo_name}/main/{path}"

        try:
            response = await client.get(url)

            if response.status_code == 200:
                return response.json()

            # Try master branch
            url = f"{GITHUB_RAW}/{owner}/{repo_name}/master/{path}"
            response = await client.get(url)

            if response.status_code == 200:
                return response.json()

        except Exception as e:
            logger.debug(f"Error fetching {path}: {e}")

    return None


def extract_dtc_codes(signalset: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract DTC codes from a signalset.

    Args:
        signalset: Signalset dictionary.

    Returns:
        List of DTC code dictionaries.
    """
    codes = []

    # Navigate to DTCs in signalset
    dtcs = signalset.get("dtcs", {})

    if not dtcs:
        # Try alternative locations - commands can be list or dict
        commands = signalset.get("commands", [])

        # Handle commands as list (v3 format)
        if isinstance(commands, list):
            for cmd_data in commands:
                if isinstance(cmd_data, dict):
                    cmd_dtcs = cmd_data.get("dtcs", {})
                    if isinstance(cmd_dtcs, dict):
                        dtcs.update(cmd_dtcs)
        # Handle commands as dict (older format)
        elif isinstance(commands, dict):
            for cmd_name, cmd_data in commands.items():
                if isinstance(cmd_data, dict) and "dtc" in cmd_name.lower():
                    cmd_dtcs = cmd_data.get("dtcs", {})
                    if isinstance(cmd_dtcs, dict):
                        dtcs.update(cmd_dtcs)

    # Handle dtcs as dict
    if isinstance(dtcs, dict):
        for code, code_data in dtcs.items():
            if not re.match(r'^[PCBU][0-9A-F]{4}$', code, re.IGNORECASE):
                continue

            description = code_data if isinstance(code_data, str) else code_data.get("description", "")

            if description:
                codes.append({
                    "code": code.upper(),
                    "description_en": description,
                    "is_generic": code[1] == "0",
                })

    return codes


def extract_signals(signalset: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract PIDs/signals from a signalset.

    Args:
        signalset: Signalset dictionary.

    Returns:
        List of signal dictionaries.
    """
    signals = []

    commands = signalset.get("commands", signalset.get("pids", []))

    # Handle commands as list (v3 format)
    if isinstance(commands, list):
        for cmd_data in commands:
            if not isinstance(cmd_data, dict):
                continue

            # v3 format has signals nested inside each command
            cmd_signals = cmd_data.get("signals", [])
            cmd_info = cmd_data.get("cmd", {})

            # Get PID from cmd (e.g., {"22": "2AB2"})
            pid = None
            if isinstance(cmd_info, dict):
                for service, pid_val in cmd_info.items():
                    pid = f"{service}:{pid_val}"
                    break

            for sig in cmd_signals:
                if not isinstance(sig, dict):
                    continue

                # Extract unit from fmt if present
                fmt = sig.get("fmt", {})
                unit = fmt.get("unit", "") if isinstance(fmt, dict) else ""

                signal = {
                    "pid": pid or sig.get("id", ""),
                    "name": sig.get("name", sig.get("id", "")),
                    "description": sig.get("description", ""),
                    "unit": unit,
                    "min_value": fmt.get("min") if isinstance(fmt, dict) else None,
                    "max_value": fmt.get("max") if isinstance(fmt, dict) else None,
                    "formula": "",
                    "signal_id": sig.get("id", ""),
                    "path": sig.get("path", ""),
                }

                signals.append(signal)

    # Handle commands as dict (older format)
    elif isinstance(commands, dict):
        for cmd_id, cmd_data in commands.items():
            if not isinstance(cmd_data, dict):
                continue

            signal = {
                "pid": cmd_id,
                "name": cmd_data.get("name", cmd_id),
                "description": cmd_data.get("description", ""),
                "unit": cmd_data.get("unit", ""),
                "min_value": cmd_data.get("min"),
                "max_value": cmd_data.get("max"),
                "formula": cmd_data.get("formula", ""),
            }

            signals.append(signal)

    return signals


async def process_repos(
    repos: List[Dict[str, Any]],
    max_repos: int = MAX_REPOS,
) -> Dict[str, Any]:
    """
    Process repositories and extract vehicle data.

    Args:
        repos: List of repository info dictionaries.
        max_repos: Maximum number of repos to process.

    Returns:
        Dictionary with vehicles and their data.
    """
    logger.info(f"Processing up to {max_repos} repositories...")

    vehicles = {}
    all_dtcs = []

    # Sort repos to prioritize European makes
    def repo_priority(repo: Dict[str, Any]) -> int:
        name = repo.get("name", "")
        for i, make in enumerate(PRIORITY_MAKES):
            if make.lower() in name.lower():
                return i
        return len(PRIORITY_MAKES) + 1

    sorted_repos = sorted(repos, key=repo_priority)[:max_repos]

    async with httpx.AsyncClient(timeout=30.0) as client:
        for repo in tqdm(sorted_repos, desc="Processing repos"):
            repo_name = repo.get("name", "")

            # Parse make/model
            make, model = parse_repo_name(repo_name)
            if not make or not model:
                continue

            # Download signalset
            signalset = await download_signalset(client, repo_name)

            if not signalset:
                continue

            # Extract data
            dtcs = extract_dtc_codes(signalset)
            signals = extract_signals(signalset)

            # Store vehicle data
            vehicle_id = f"{make.lower().replace(' ', '_')}_{model.lower().replace(' ', '_')}"

            vehicles[vehicle_id] = {
                "id": vehicle_id,
                "make": make,
                "model": model,
                "repo": repo_name,
                "dtc_count": len(dtcs),
                "signal_count": len(signals),
                "dtcs": dtcs,
                "signals": signals,
            }

            # Add DTCs to global list with vehicle reference (use copy to avoid mutation!)
            for dtc in dtcs:
                dtc_copy = dtc.copy()  # Avoid mutating original dict
                dtc_copy["vehicle_id"] = vehicle_id
                dtc_copy["make"] = make
                dtc_copy["model"] = model
                all_dtcs.append(dtc_copy)

            await asyncio.sleep(RATE_LIMIT_DELAY)

    logger.info(f"Processed {len(vehicles)} vehicles with {len(all_dtcs)} total DTCs")

    return {
        "vehicles": vehicles,
        "dtcs": all_dtcs,
        "metadata": {
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "vehicle_count": len(vehicles),
            "dtc_count": len(all_dtcs),
        }
    }


def save_to_cache(data: Dict[str, Any], file_path: Path) -> None:
    """Save data to cache file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved to {file_path}")


def load_from_cache(file_path: Path) -> Optional[Dict[str, Any]]:
    """Load data from cache file."""
    if not file_path.exists():
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_sync_db_url() -> str:
    """Convert async database URL to sync."""
    from backend.app.core.config import settings
    url = settings.DATABASE_URL
    if url.startswith("postgresql+asyncpg://"):
        url = url.replace("postgresql+asyncpg://", "postgresql://")
    return url


def import_vehicles_to_postgres(vehicles: Dict[str, Dict[str, Any]]) -> int:
    """
    Import vehicles to PostgreSQL.

    Args:
        vehicles: Dictionary of vehicle data.

    Returns:
        Number of imported vehicles.
    """
    from backend.app.db.postgres.models import Base, VehicleMake, VehicleModel
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    logger.info("Importing vehicles to PostgreSQL...")

    db_url = get_sync_db_url()
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)

    imported_makes = 0
    imported_models = 0

    with Session(engine) as session:
        # Group by make
        makes = {}
        for vehicle_id, vehicle in vehicles.items():
            make = vehicle.get("make", "")
            if make not in makes:
                makes[make] = []
            makes[make].append(vehicle)

        # Import makes and models
        for make_name, make_vehicles in makes.items():
            make_id = make_name.lower().replace(" ", "_").replace("-", "_")

            # Check if make exists
            existing_make = session.query(VehicleMake).filter_by(id=make_id).first()

            if not existing_make:
                make = VehicleMake(
                    id=make_id,
                    name=make_name,
                )
                session.add(make)
                imported_makes += 1

            # Import models
            for vehicle in make_vehicles:
                model_name = vehicle.get("model", "")
                model_id = f"{make_id}_{model_name.lower().replace(' ', '_').replace('-', '_')}"

                existing_model = session.query(VehicleModel).filter_by(id=model_id).first()

                if not existing_model:
                    model = VehicleModel(
                        id=model_id,
                        name=model_name,
                        make_id=make_id,
                        year_start=1990,  # Default, OBDb doesn't always have years
                    )
                    session.add(model)
                    imported_models += 1

        session.commit()

    logger.info(f"Imported {imported_makes} makes and {imported_models} models")
    return imported_models


def import_dtcs_to_postgres(dtcs: List[Dict[str, Any]], batch_size: int = 100) -> int:
    """
    Import vehicle-specific DTCs to PostgreSQL.

    Args:
        dtcs: List of DTC dictionaries with vehicle references.
        batch_size: Batch size for inserts.

    Returns:
        Number of new DTCs imported.
    """
    from backend.app.db.postgres.models import Base, DTCCode
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    logger.info("Importing DTCs to PostgreSQL...")

    db_url = get_sync_db_url()
    engine = create_engine(db_url)

    imported = 0

    with Session(engine) as session:
        for i in tqdm(range(0, len(dtcs), batch_size), desc="Importing DTCs"):
            batch = dtcs[i:i + batch_size]

            for dtc in batch:
                code = dtc.get("code", "")

                existing = session.query(DTCCode).filter_by(code=code).first()

                if existing:
                    # Add vehicle to applicable_makes if not already there
                    make = dtc.get("make", "")
                    if make and make not in (existing.applicable_makes or []):
                        if existing.applicable_makes is None:
                            existing.applicable_makes = []
                        existing.applicable_makes = existing.applicable_makes + [make]
                else:
                    # Create new DTC
                    new_dtc = DTCCode(
                        code=code,
                        description_en=dtc.get("description_en", ""),
                        category=dtc.get("category", "powertrain"),
                        severity="medium",
                        is_generic=dtc.get("is_generic", False),
                        applicable_makes=[dtc.get("make", "")] if dtc.get("make") else [],
                    )
                    session.add(new_dtc)
                    imported += 1

            session.commit()

    logger.info(f"Imported {imported} new DTCs")
    return imported


def import_to_neo4j(vehicles: Dict[str, Dict[str, Any]], dtcs: List[Dict[str, Any]]) -> int:
    """
    Import vehicle data to Neo4j.

    Args:
        vehicles: Dictionary of vehicle data.
        dtcs: List of DTC dictionaries.

    Returns:
        Number of nodes created.
    """
    from backend.app.db.neo4j_models import DTCNode

    logger.info("Importing to Neo4j...")

    # Import DTCs
    created = 0

    for dtc in tqdm(dtcs, desc="Neo4j import"):
        code = dtc.get("code", "")

        existing = DTCNode.nodes.get_or_none(code=code)

        if not existing:
            DTCNode(
                code=code,
                description_en=dtc.get("description_en", ""),
                category=dtc.get("category", "powertrain"),
                severity="medium",
                is_generic=bool(dtc.get("is_generic", False)),  # Keep as boolean!
            ).save()
            created += 1

    logger.info(f"Created {created} Neo4j nodes")
    return created


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Import vehicle data from OBDb GitHub organization"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available repositories",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download data from OBDb repos",
    )
    parser.add_argument(
        "--postgres",
        action="store_true",
        help="Import to PostgreSQL",
    )
    parser.add_argument(
        "--neo4j",
        action="store_true",
        help="Import to Neo4j",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download and import to all databases",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use cached data instead of downloading",
    )
    parser.add_argument(
        "--max-repos",
        type=int,
        default=MAX_REPOS,
        help=f"Maximum repos to process (default: {MAX_REPOS})",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Ensure cache directory exists
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # List repos
        if args.list:
            async with httpx.AsyncClient(timeout=30.0) as client:
                repos = await list_obdb_repos(client)
                save_to_cache({"repos": repos}, REPOS_CACHE)

                print("\n" + "=" * 60)
                print("OBDb REPOSITORIES")
                print("=" * 60)

                for repo in repos[:50]:
                    name = repo.get("name", "")
                    make, model = parse_repo_name(name)
                    if make and model:
                        print(f"  {make} {model}")

                if len(repos) > 50:
                    print(f"  ... and {len(repos) - 50} more")

                print(f"\nTotal: {len(repos)} repositories")
                print("=" * 60)
            return

        # Get vehicle data
        if args.use_cache and VEHICLES_CACHE.exists():
            data = load_from_cache(VEHICLES_CACHE)
        elif args.download or args.all:
            # First get repo list
            async with httpx.AsyncClient(timeout=30.0) as client:
                repos = await list_obdb_repos(client)

            # Process repos
            data = await process_repos(repos, max_repos=args.max_repos)
            save_to_cache(data, VEHICLES_CACHE)
        else:
            data = load_from_cache(VEHICLES_CACHE)

        if not data:
            logger.error("No data available. Use --download first.")
            sys.exit(1)

        vehicles = data.get("vehicles", {})
        dtcs = data.get("dtcs", [])

        logger.info(f"Loaded {len(vehicles)} vehicles with {len(dtcs)} DTCs")

        # Import to databases
        if args.postgres or args.all:
            import_vehicles_to_postgres(vehicles)
            import_dtcs_to_postgres(dtcs)

        if args.neo4j or args.all:
            import_to_neo4j(vehicles, dtcs)

        # Print summary
        print("\n" + "=" * 60)
        print("OBDb IMPORT SUMMARY")
        print("=" * 60)
        print(f"Vehicles processed: {len(vehicles)}")
        print(f"DTCs found: {len(dtcs)}")

        # Make breakdown
        makes = {}
        for vehicle in vehicles.values():
            make = vehicle.get("make", "Unknown")
            makes[make] = makes.get(make, 0) + 1

        print("\nTop makes:")
        for make, count in sorted(makes.items(), key=lambda x: -x[1])[:10]:
            print(f"  {make}: {count} models")
        print("=" * 60)

    except Exception as e:
        logger.error(f"Import failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
