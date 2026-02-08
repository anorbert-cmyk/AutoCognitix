#!/usr/bin/env python3
"""
OBDb GitHub Repository Full Importer

Imports ALL vehicle data from the OBDb (Open Source Diagnostic Database) GitHub organization.
Organizes data into make/model directory structure with comprehensive statistics.

GitHub org: https://github.com/OBDb
Structure: Each vehicle has its own repo (e.g., https://github.com/OBDb/Toyota-Camry)

Features:
- Async parallel downloads with configurable concurrency
- GitHub rate limit handling (60/hour unauthenticated, 5000/hour with token)
- Progress bar with real-time status
- Resume capability (skips already downloaded)
- make/model directory organization
- Comprehensive statistics and reporting

Usage:
    python scripts/import_obdb_github.py              # Full import
    python scripts/import_obdb_github.py --list       # List repos only
    python scripts/import_obdb_github.py --stats      # Show statistics
    python scripts/import_obdb_github.py --reorganize # Reorganize existing data into make/model

Environment:
    GITHUB_TOKEN - Optional GitHub token for higher rate limits
"""

import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import aiofiles
import httpx
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "obdb"
OUTPUT_DIR = DATA_DIR / "vehicles"  # New structure: data/obdb/vehicles/{make}/{model}/
LEGACY_DIR = DATA_DIR / "signalsets"  # Old flat structure
METADATA_DIR = DATA_DIR / "metadata"
REPOS_CACHE = METADATA_DIR / "repos_cache.json"
IMPORT_STATE_FILE = METADATA_DIR / "import_state.json"
STATS_FILE = DATA_DIR / "import_stats.json"

# GitHub API configuration
GITHUB_API = "https://api.github.com"
GITHUB_RAW = "https://raw.githubusercontent.com"
OBDB_ORG = "OBDb"

# Rate limiting and concurrency settings
DEFAULT_CONCURRENCY = 10
API_RATE_LIMIT_DELAY = 0.5  # Seconds between API calls
RAW_RATE_LIMIT_DELAY = 0.1  # Seconds between raw file downloads
HTTP_TIMEOUT = 30.0


@dataclass
class ImportState:
    """Tracks import progress for resume capability."""

    downloaded: Set[str] = field(default_factory=set)
    failed: Set[str] = field(default_factory=set)
    skipped: Set[str] = field(default_factory=set)
    last_updated: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "downloaded": list(self.downloaded),
            "failed": list(self.failed),
            "skipped": list(self.skipped),
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImportState":
        return cls(
            downloaded=set(data.get("downloaded", [])),
            failed=set(data.get("failed", [])),
            skipped=set(data.get("skipped", [])),
            last_updated=data.get("last_updated", ""),
        )


@dataclass
class RepoInfo:
    """Information about an OBDb repository."""

    name: str
    full_name: str
    html_url: str
    description: str = ""
    make: str = ""
    model: str = ""
    is_vehicle: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "full_name": self.full_name,
            "html_url": self.html_url,
            "description": self.description,
            "make": self.make,
            "model": self.model,
            "is_vehicle": self.is_vehicle,
        }


@dataclass
class VehicleData:
    """Parsed vehicle data from signalset."""

    repo_name: str
    make: str
    model: str
    signalset: Optional[Dict[str, Any]] = None
    readme: str = ""
    signal_count: int = 0
    dtc_count: int = 0
    command_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "repo_name": self.repo_name,
            "make": self.make,
            "model": self.model,
            "signal_count": self.signal_count,
            "dtc_count": self.dtc_count,
            "command_count": self.command_count,
        }


# Special make names that contain hyphens
SPECIAL_MAKES = {
    "Mercedes-Benz": "Mercedes-Benz",
    "Alfa-Romeo": "Alfa Romeo",
    "Land-Rover": "Land Rover",
    "Aston-Martin": "Aston Martin",
    "Rolls-Royce": "Rolls-Royce",
}

# Non-vehicle repositories to skip
NON_VEHICLE_REPOS = {
    ".github",
    ".meta",
    ".vehicle-template",
    ".schemas",
    ".make",
    ".claude",
    ".devcontainer",
    "obdb.community",
    "editor.obdb.community",
    "logformatter.obdb.community",
    "pidhunter.obdb.community",
    "SAEJ1979",
    "vscode",
}


def parse_repo_name(name: str) -> Tuple[str, str, bool]:
    """
    Parse make and model from repository name.

    Args:
        name: Repository name (e.g., "Toyota-Camry", "Mercedes-Benz-S-Class")

    Returns:
        Tuple of (make, model, is_vehicle)
    """
    # Skip non-vehicle repos
    if name in NON_VEHICLE_REPOS or name.startswith("."):
        return name, "", False

    # Handle special makes with hyphens
    for special, replacement in SPECIAL_MAKES.items():
        if name.startswith(special + "-"):
            model = name[len(special) + 1 :].replace("-", " ")
            return replacement, model, True

    # Standard parsing: first segment is make
    parts = name.split("-", 1)
    if len(parts) == 2:
        make = parts[0].replace("_", " ")
        model = parts[1].replace("-", " ").replace("_", " ")
        return make, model, True

    return name, "", False


def get_github_headers() -> Dict[str, str]:
    """Get GitHub API headers with optional token authentication."""
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "AutoCognitix-OBDb-Importer/1.0",
    }

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
        logger.info("Using GitHub token authentication (5000 requests/hour)")
    else:
        logger.warning("No GITHUB_TOKEN found. Rate limited to 60 requests/hour.")
        logger.warning("Set GITHUB_TOKEN environment variable for higher limits.")

    return headers


async def fetch_all_repos(client: httpx.AsyncClient) -> List[RepoInfo]:
    """
    Fetch all repositories from OBDb organization using pagination.

    Returns:
        List of RepoInfo objects for all vehicle repos.
    """
    repos = []
    page = 1
    per_page = 100
    headers = get_github_headers()

    logger.info("Fetching repository list from GitHub API...")

    while True:
        url = f"{GITHUB_API}/orgs/{OBDB_ORG}/repos"
        params = {"page": page, "per_page": per_page, "type": "public"}

        try:
            response = await client.get(url, headers=headers, params=params)

            # Check rate limit
            remaining = response.headers.get("X-RateLimit-Remaining", "?")
            reset_time = response.headers.get("X-RateLimit-Reset", "")

            if response.status_code == 403:
                reset_dt = (
                    datetime.fromtimestamp(int(reset_time), tz=timezone.utc)
                    if reset_time
                    else "unknown"
                )
                logger.error(f"Rate limit exceeded. Resets at: {reset_dt}")
                logger.error(
                    "Set GITHUB_TOKEN environment variable for 5000 requests/hour"
                )
                break

            if response.status_code != 200:
                logger.error(f"GitHub API error: {response.status_code}")
                break

            page_repos = response.json()

            if not page_repos:
                break

            for repo_data in page_repos:
                name = repo_data.get("name", "")
                make, model, is_vehicle = parse_repo_name(name)

                repo = RepoInfo(
                    name=name,
                    full_name=repo_data.get("full_name", ""),
                    html_url=repo_data.get("html_url", ""),
                    description=repo_data.get("description", "") or "",
                    make=make,
                    model=model,
                    is_vehicle=is_vehicle,
                )
                repos.append(repo)

            logger.info(
                f"Page {page}: {len(page_repos)} repos (API rate limit: {remaining} remaining)"
            )

            page += 1
            await asyncio.sleep(API_RATE_LIMIT_DELAY)

        except httpx.TimeoutException:
            logger.error(f"Timeout fetching page {page}")
            break
        except Exception as e:
            logger.error(f"Error fetching repos: {e}")
            break

    # Filter to vehicle repos only
    vehicle_repos = [r for r in repos if r.is_vehicle]
    logger.info(f"Total repositories: {len(repos)}, Vehicle repos: {len(vehicle_repos)}")

    return vehicle_repos


async def download_vehicle_data(
    client: httpx.AsyncClient,
    repo: RepoInfo,
    semaphore: asyncio.Semaphore,
) -> Tuple[str, Optional[VehicleData], str]:
    """
    Download signalset and README from a repository.

    Args:
        client: HTTP client
        repo: Repository info
        semaphore: Concurrency limiter

    Returns:
        Tuple of (repo_name, VehicleData or None, error message or "")
    """
    async with semaphore:
        vehicle = VehicleData(
            repo_name=repo.name,
            make=repo.make,
            model=repo.model,
        )

        # Try different signalset paths and branches
        signalset_paths = [
            ("main", "signalsets/v3/default.json"),
            ("master", "signalsets/v3/default.json"),
            ("main", "signalsets/v2/default.json"),
            ("master", "signalsets/v2/default.json"),
            ("main", "signalsets/default.json"),
            ("master", "signalsets/default.json"),
        ]

        signalset_found = False
        for branch, path in signalset_paths:
            url = f"{GITHUB_RAW}/{OBDB_ORG}/{repo.name}/{branch}/{path}"
            try:
                response = await client.get(url)
                if response.status_code == 200:
                    try:
                        vehicle.signalset = response.json()
                        signalset_found = True

                        # Count signals and commands
                        commands = vehicle.signalset.get("commands", [])
                        if isinstance(commands, list):
                            vehicle.command_count = len(commands)
                            for cmd in commands:
                                if isinstance(cmd, dict):
                                    signals = cmd.get("signals", [])
                                    vehicle.signal_count += (
                                        len(signals) if isinstance(signals, list) else 0
                                    )
                        elif isinstance(commands, dict):
                            vehicle.command_count = len(commands)
                            vehicle.signal_count = len(commands)

                        # Count DTCs
                        dtcs = vehicle.signalset.get("dtcs", {})
                        if isinstance(dtcs, dict):
                            # Count valid DTC codes
                            for code in dtcs:
                                if re.match(r"^[PCBU][0-9A-F]{4}$", code, re.IGNORECASE):
                                    vehicle.dtc_count += 1

                        break
                    except json.JSONDecodeError:
                        continue
            except (httpx.TimeoutException, httpx.RequestError):
                continue

        if not signalset_found:
            return repo.name, None, "No signalset found"

        # Try to download README
        for branch in ["main", "master"]:
            readme_url = f"{GITHUB_RAW}/{OBDB_ORG}/{repo.name}/{branch}/README.md"
            try:
                response = await client.get(readme_url)
                if response.status_code == 200:
                    vehicle.readme = response.text
                    break
            except (httpx.TimeoutException, httpx.RequestError):
                continue

        await asyncio.sleep(RAW_RATE_LIMIT_DELAY)
        return repo.name, vehicle, ""


async def save_vehicle_data(vehicle: VehicleData) -> None:
    """Save vehicle data to make/model directory structure."""
    # Sanitize make and model for filesystem
    safe_make = re.sub(r'[<>:"/\\|?*]', "_", vehicle.make)
    safe_model = re.sub(r'[<>:"/\\|?*]', "_", vehicle.model)

    vehicle_dir = OUTPUT_DIR / safe_make / safe_model
    vehicle_dir.mkdir(parents=True, exist_ok=True)

    # Save signalset
    if vehicle.signalset:
        signalset_path = vehicle_dir / "signalset.json"
        async with aiofiles.open(signalset_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(vehicle.signalset, ensure_ascii=False, indent=2))

    # Save README
    if vehicle.readme:
        readme_path = vehicle_dir / "README.md"
        async with aiofiles.open(readme_path, "w", encoding="utf-8") as f:
            await f.write(vehicle.readme)

    # Save metadata
    metadata_path = vehicle_dir / "metadata.json"
    metadata = {
        "repo_name": vehicle.repo_name,
        "make": vehicle.make,
        "model": vehicle.model,
        "signal_count": vehicle.signal_count,
        "dtc_count": vehicle.dtc_count,
        "command_count": vehicle.command_count,
        "github_url": f"https://github.com/OBDb/{vehicle.repo_name}",
        "imported_at": datetime.now(timezone.utc).isoformat(),
    }
    async with aiofiles.open(metadata_path, "w", encoding="utf-8") as f:
        await f.write(json.dumps(metadata, ensure_ascii=False, indent=2))


def load_import_state() -> ImportState:
    """Load import state from file."""
    if IMPORT_STATE_FILE.exists():
        with open(IMPORT_STATE_FILE, encoding="utf-8") as f:
            return ImportState.from_dict(json.load(f))
    return ImportState()


def save_import_state(state: ImportState) -> None:
    """Save import state to file."""
    state.last_updated = datetime.now(timezone.utc).isoformat()
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(IMPORT_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state.to_dict(), f, indent=2)


async def import_all_vehicles(
    repos: List[RepoInfo],
    concurrency: int = DEFAULT_CONCURRENCY,
    force: bool = False,
) -> ImportState:
    """
    Import all vehicle data from OBDb repositories.

    Args:
        repos: List of repositories to import
        concurrency: Number of parallel downloads
        force: Force re-download even if already imported

    Returns:
        Updated import state
    """
    state = load_import_state() if not force else ImportState()

    # Filter repos to import
    to_import = []
    for repo in repos:
        if repo.name in state.downloaded and not force:
            continue
        to_import.append(repo)

    if not to_import:
        logger.info("All vehicles already imported!")
        return state

    logger.info(
        f"Importing {len(to_import)} vehicles ({len(state.downloaded)} already done)"
    )

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(concurrency)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        tasks = [
            download_vehicle_data(client, repo, semaphore) for repo in to_import
        ]

        # Process with progress bar
        results = await tqdm_asyncio.gather(*tasks, desc="Importing vehicles")

        # Process results
        for repo_name, vehicle, error in results:
            if vehicle:
                await save_vehicle_data(vehicle)
                state.downloaded.add(repo_name)
                state.failed.discard(repo_name)
            elif error == "No signalset found":
                state.skipped.add(repo_name)
                state.failed.discard(repo_name)
            else:
                state.failed.add(repo_name)
                logger.warning(f"{repo_name}: {error}")

        # Save state
        save_import_state(state)

    return state


def reorganize_existing_data() -> Dict[str, Any]:
    """
    Reorganize existing signalsets from flat structure to make/model structure.

    Returns:
        Statistics about the reorganization
    """
    if not LEGACY_DIR.exists():
        logger.warning(f"Legacy directory not found: {LEGACY_DIR}")
        return {"reorganized": 0, "skipped": 0, "errors": []}

    logger.info("Reorganizing existing signalsets...")

    stats = {"reorganized": 0, "skipped": 0, "errors": []}

    signalset_files = list(LEGACY_DIR.glob("*.json"))

    for file_path in tqdm(signalset_files, desc="Reorganizing"):
        try:
            repo_name = file_path.stem

            # Skip template files
            if repo_name.startswith("."):
                stats["skipped"] += 1
                continue

            make, model, is_vehicle = parse_repo_name(repo_name)

            if not is_vehicle or not model:
                stats["skipped"] += 1
                continue

            # Sanitize for filesystem
            safe_make = re.sub(r'[<>:"/\\|?*]', "_", make)
            safe_model = re.sub(r'[<>:"/\\|?*]', "_", model)

            vehicle_dir = OUTPUT_DIR / safe_make / safe_model
            vehicle_dir.mkdir(parents=True, exist_ok=True)

            # Copy signalset
            dest_path = vehicle_dir / "signalset.json"
            if not dest_path.exists():
                shutil.copy2(file_path, dest_path)

                # Load and create metadata
                with open(file_path, encoding="utf-8") as f:
                    signalset = json.load(f)

                # Count signals
                signal_count = 0
                command_count = 0
                dtc_count = 0

                commands = signalset.get("commands", [])
                if isinstance(commands, list):
                    command_count = len(commands)
                    for cmd in commands:
                        if isinstance(cmd, dict):
                            signals = cmd.get("signals", [])
                            signal_count += len(signals) if isinstance(signals, list) else 0
                elif isinstance(commands, dict):
                    command_count = len(commands)
                    signal_count = len(commands)

                dtcs = signalset.get("dtcs", {})
                if isinstance(dtcs, dict):
                    for code in dtcs:
                        if re.match(r"^[PCBU][0-9A-F]{4}$", code, re.IGNORECASE):
                            dtc_count += 1

                # Save metadata
                metadata_path = vehicle_dir / "metadata.json"
                metadata = {
                    "repo_name": repo_name,
                    "make": make,
                    "model": model,
                    "signal_count": signal_count,
                    "dtc_count": dtc_count,
                    "command_count": command_count,
                    "github_url": f"https://github.com/OBDb/{repo_name}",
                    "reorganized_at": datetime.now(timezone.utc).isoformat(),
                }
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)

                stats["reorganized"] += 1
            else:
                stats["skipped"] += 1

        except Exception as e:
            stats["errors"].append({"file": str(file_path), "error": str(e)})

    logger.info(
        f"Reorganized {stats['reorganized']} vehicles, "
        f"skipped {stats['skipped']}, errors: {len(stats['errors'])}"
    )

    return stats


def generate_statistics() -> Dict[str, Any]:
    """Generate comprehensive statistics about imported data."""
    logger.info("Generating statistics...")

    stats = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_repos": 0,
        "total_vehicles": 0,
        "total_makes": 0,
        "total_signals": 0,
        "total_dtcs": 0,
        "total_commands": 0,
        "makes": {},
        "top_vehicles_by_signals": [],
        "vehicles_with_dtcs": [],
    }

    if not OUTPUT_DIR.exists():
        logger.warning("Output directory not found. Run import first.")
        return stats

    vehicles_data = []

    # Scan make directories
    make_dirs = [d for d in OUTPUT_DIR.iterdir() if d.is_dir()]
    stats["total_makes"] = len(make_dirs)

    for make_dir in tqdm(make_dirs, desc="Scanning vehicles"):
        make_name = make_dir.name
        make_stats = {
            "model_count": 0,
            "signal_count": 0,
            "dtc_count": 0,
            "models": [],
        }

        # Scan model directories
        model_dirs = [d for d in make_dir.iterdir() if d.is_dir()]

        for model_dir in model_dirs:
            metadata_path = model_dir / "metadata.json"

            if metadata_path.exists():
                with open(metadata_path, encoding="utf-8") as f:
                    metadata = json.load(f)

                make_stats["model_count"] += 1
                make_stats["signal_count"] += metadata.get("signal_count", 0)
                make_stats["dtc_count"] += metadata.get("dtc_count", 0)
                make_stats["models"].append(metadata.get("model", model_dir.name))

                stats["total_vehicles"] += 1
                stats["total_signals"] += metadata.get("signal_count", 0)
                stats["total_dtcs"] += metadata.get("dtc_count", 0)
                stats["total_commands"] += metadata.get("command_count", 0)

                vehicles_data.append(
                    {
                        "make": make_name,
                        "model": metadata.get("model", model_dir.name),
                        "signals": metadata.get("signal_count", 0),
                        "dtcs": metadata.get("dtc_count", 0),
                        "commands": metadata.get("command_count", 0),
                    }
                )

        stats["makes"][make_name] = make_stats

    # Sort makes by vehicle count
    stats["makes"] = dict(
        sorted(stats["makes"].items(), key=lambda x: -x[1]["model_count"])
    )

    # Top vehicles by signals
    vehicles_data.sort(key=lambda x: -x["signals"])
    stats["top_vehicles_by_signals"] = vehicles_data[:20]

    # Vehicles with DTCs
    vehicles_with_dtcs = [v for v in vehicles_data if v["dtcs"] > 0]
    vehicles_with_dtcs.sort(key=lambda x: -x["dtcs"])
    stats["vehicles_with_dtcs"] = vehicles_with_dtcs[:20]

    # Load repo count from cache if available
    if REPOS_CACHE.exists():
        with open(REPOS_CACHE, encoding="utf-8") as f:
            repos = json.load(f)
            stats["total_repos"] = len(repos)

    # Save stats
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    return stats


def print_statistics(stats: Dict[str, Any]) -> None:
    """Print formatted statistics."""
    print("\n" + "=" * 70)
    print("OBDB IMPORT STATISTICS")
    print("=" * 70)
    print(f"Generated: {stats['generated_at']}")
    print("-" * 70)
    print(f"Total GitHub repos:     {stats['total_repos']}")
    print(f"Total vehicles:         {stats['total_vehicles']}")
    print(f"Total makes:            {stats['total_makes']}")
    print(f"Total signals/PIDs:     {stats['total_signals']:,}")
    print(f"Total DTC codes:        {stats['total_dtcs']:,}")
    print(f"Total commands:         {stats['total_commands']:,}")
    print("-" * 70)
    print("TOP 15 MAKES BY VEHICLE COUNT:")
    makes = stats.get("makes", {})
    for i, (make, data) in enumerate(list(makes.items())[:15]):
        print(f"  {make:25} {data['model_count']:4} models, {data['signal_count']:5} signals")
    if len(makes) > 15:
        print(f"  ... and {len(makes) - 15} more makes")
    print("-" * 70)
    print("TOP 10 VEHICLES BY SIGNAL COUNT:")
    for v in stats.get("top_vehicles_by_signals", [])[:10]:
        print(f"  {v['make']:15} {v['model']:25} {v['signals']:5} signals")
    print("-" * 70)
    if stats.get("vehicles_with_dtcs"):
        print("VEHICLES WITH DTC CODES:")
        for v in stats.get("vehicles_with_dtcs", [])[:5]:
            print(f"  {v['make']:15} {v['model']:25} {v['dtcs']:5} DTCs")
    print("=" * 70)


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Import ALL OBDb vehicle data from GitHub"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all repositories without downloading",
    )
    parser.add_argument(
        "--import",
        dest="do_import",
        action="store_true",
        help="Import all vehicle data (default action)",
    )
    parser.add_argument(
        "--reorganize",
        action="store_true",
        help="Reorganize existing data into make/model structure",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show statistics only",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-import even if already done",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Number of parallel downloads (default: {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Ensure directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    # Stats only mode
    if args.stats:
        stats = generate_statistics()
        print_statistics(stats)
        return

    # Reorganize mode
    if args.reorganize:
        reorg_stats = reorganize_existing_data()
        print(f"\nReorganization complete: {reorg_stats['reorganized']} vehicles")
        stats = generate_statistics()
        print_statistics(stats)
        return

    # Load or fetch repos
    repos = []
    if REPOS_CACHE.exists() and not args.force:
        logger.info("Loading cached repository list...")
        with open(REPOS_CACHE, encoding="utf-8") as f:
            repos_data = json.load(f)
            repos = [RepoInfo(**r) for r in repos_data]

    if not repos or args.list or args.force:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            repos = await fetch_all_repos(client)

        # Save repos cache
        with open(REPOS_CACHE, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in repos], f, indent=2)

    # List only mode
    if args.list:
        print(f"\nFound {len(repos)} vehicle repositories:\n")

        # Group by make
        makes: Dict[str, List[str]] = {}
        for repo in repos:
            if repo.make not in makes:
                makes[repo.make] = []
            makes[repo.make].append(repo.model)

        # Sort by count
        sorted_makes = sorted(makes.items(), key=lambda x: -len(x[1]))

        for make, models in sorted_makes[:20]:
            print(f"  {make} ({len(models)} models):")
            for model in sorted(models)[:5]:
                print(f"    - {model}")
            if len(models) > 5:
                print(f"    ... and {len(models) - 5} more")
        if len(sorted_makes) > 20:
            print(f"\n  ... and {len(sorted_makes) - 20} more makes")
        return

    # Default: Import all
    state = await import_all_vehicles(
        repos, concurrency=args.concurrency, force=args.force
    )

    # Generate and show statistics
    stats = generate_statistics()
    print_statistics(stats)


if __name__ == "__main__":
    asyncio.run(main())
