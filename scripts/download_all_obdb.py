#!/usr/bin/env python3
"""
OBDb Full Repository Downloader

Downloads ALL signalsets from the OBDb GitHub organization with:
- Parallel downloads (configurable concurrency)
- Resume capability (skips already downloaded)
- Rate limiting (respects GitHub API limits)
- Progress tracking
- Comprehensive reporting

Usage:
    python scripts/download_all_obdb.py              # Download all
    python scripts/download_all_obdb.py --list       # List repos only
    python scripts/download_all_obdb.py --stats      # Show statistics
    python scripts/download_all_obdb.py --parse      # Parse downloaded files
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
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
SIGNALSETS_DIR = DATA_DIR / "signalsets"
METADATA_DIR = DATA_DIR / "metadata"
REPOS_FILE = METADATA_DIR / "all_repos.json"
DOWNLOAD_STATE_FILE = METADATA_DIR / "download_state.json"
PARSED_DATA_FILE = DATA_DIR / "parsed_vehicles.json"
SUMMARY_FILE = DATA_DIR / "download_summary.json"

# GitHub configuration
GITHUB_API = "https://api.github.com"
GITHUB_RAW = "https://raw.githubusercontent.com"
OBDB_ORG = "OBDb"

# Download settings
DEFAULT_CONCURRENCY = 10  # Parallel downloads
RATE_LIMIT_DELAY = 0.1    # Seconds between requests (within concurrency)
API_RATE_LIMIT_DELAY = 0.5  # Seconds between API calls
TIMEOUT = 30.0            # HTTP timeout


@dataclass
class DownloadState:
    """Tracks download progress for resume capability."""
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
    def from_dict(cls, data: Dict[str, Any]) -> "DownloadState":
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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ParsedVehicle:
    """Parsed vehicle data from signalset."""
    repo_name: str
    make: str
    model: str
    signals: List[Dict[str, Any]] = field(default_factory=list)
    dtcs: List[Dict[str, Any]] = field(default_factory=list)
    raw_commands_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "repo_name": self.repo_name,
            "make": self.make,
            "model": self.model,
            "signal_count": len(self.signals),
            "dtc_count": len(self.dtcs),
            "raw_commands_count": self.raw_commands_count,
            "signals": self.signals,
            "dtcs": self.dtcs,
        }


def get_github_headers() -> Dict[str, str]:
    """Get GitHub API headers with optional token."""
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "AutoCognitix-OBDb-Downloader/1.0",
    }

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
        logger.info("Using GitHub token for authentication (5000 req/hour)")
    else:
        logger.warning("No GitHub token found. Rate limited to 60 req/hour.")
        logger.warning("Set GITHUB_TOKEN environment variable for higher limits.")

    return headers


def parse_repo_name(name: str) -> Tuple[str, str]:
    """
    Parse make and model from repository name.

    OBDb format: "Make-Model" (e.g., "Volkswagen-Golf", "BMW-3-Series")
    """
    # Handle special cases
    special_makes = {
        "Mercedes-Benz": "Mercedes-Benz",
        "Alfa-Romeo": "Alfa Romeo",
        "Land-Rover": "Land Rover",
        "Aston-Martin": "Aston Martin",
        "Rolls-Royce": "Rolls-Royce",
    }

    for special, replacement in special_makes.items():
        if name.startswith(special + "-"):
            model = name[len(special) + 1:].replace("-", " ")
            return replacement, model

    # Standard parsing: first segment is make
    parts = name.split("-", 1)
    if len(parts) == 2:
        make = parts[0].replace("_", " ")
        model = parts[1].replace("-", " ").replace("_", " ")
        return make, model

    return name, ""


async def fetch_all_repos(client: httpx.AsyncClient) -> List[RepoInfo]:
    """
    Fetch all repositories from OBDb organization using pagination.

    Returns:
        List of RepoInfo objects for all repos.
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
                reset_dt = datetime.fromtimestamp(int(reset_time)) if reset_time else "unknown"
                logger.error(f"Rate limit exceeded. Resets at: {reset_dt}")
                break

            if response.status_code != 200:
                logger.error(f"GitHub API error: {response.status_code} - {response.text}")
                break

            page_repos = response.json()

            if not page_repos:
                break

            # Parse repos
            for repo_data in page_repos:
                name = repo_data.get("name", "")
                make, model = parse_repo_name(name)

                repo = RepoInfo(
                    name=name,
                    full_name=repo_data.get("full_name", ""),
                    html_url=repo_data.get("html_url", ""),
                    description=repo_data.get("description", "") or "",
                    make=make,
                    model=model,
                )
                repos.append(repo)

            logger.info(f"Page {page}: {len(page_repos)} repos (Rate limit: {remaining} remaining)")

            page += 1
            await asyncio.sleep(API_RATE_LIMIT_DELAY)

        except httpx.TimeoutException:
            logger.error(f"Timeout fetching page {page}")
            break
        except Exception as e:
            logger.error(f"Error fetching repos: {e}")
            break

    logger.info(f"Total repositories found: {len(repos)}")
    return repos


async def download_signalset(
    client: httpx.AsyncClient,
    repo_name: str,
    semaphore: asyncio.Semaphore,
) -> Tuple[str, Optional[Dict[str, Any]], str]:
    """
    Download signalset JSON from a repository.

    Args:
        client: HTTP client
        repo_name: Repository name
        semaphore: Concurrency limiter

    Returns:
        Tuple of (repo_name, data or None, error message or "")
    """
    async with semaphore:
        # Try different file paths and branches
        paths = [
            ("main", "signalsets/v3/default.json"),
            ("master", "signalsets/v3/default.json"),
            ("main", "signalsets/v2/default.json"),
            ("master", "signalsets/v2/default.json"),
            ("main", "signalsets/default.json"),
            ("master", "signalsets/default.json"),
        ]

        for branch, path in paths:
            url = f"{GITHUB_RAW}/{OBDB_ORG}/{repo_name}/{branch}/{path}"

            try:
                response = await client.get(url)

                if response.status_code == 200:
                    try:
                        data = response.json()
                        await asyncio.sleep(RATE_LIMIT_DELAY)
                        return repo_name, data, ""
                    except json.JSONDecodeError as e:
                        return repo_name, None, f"Invalid JSON: {e}"

            except httpx.TimeoutException:
                continue
            except Exception as e:
                continue

        return repo_name, None, "No signalset found"


async def save_signalset(repo_name: str, data: Dict[str, Any]) -> None:
    """Save signalset to file."""
    file_path = SIGNALSETS_DIR / f"{repo_name}.json"

    async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
        await f.write(json.dumps(data, ensure_ascii=False, indent=2))


def load_download_state() -> DownloadState:
    """Load download state from file."""
    if DOWNLOAD_STATE_FILE.exists():
        with open(DOWNLOAD_STATE_FILE, "r", encoding="utf-8") as f:
            return DownloadState.from_dict(json.load(f))
    return DownloadState()


def save_download_state(state: DownloadState) -> None:
    """Save download state to file."""
    state.last_updated = datetime.now(timezone.utc).isoformat()
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(DOWNLOAD_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state.to_dict(), f, indent=2)


def extract_signals(signalset: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract signals/PIDs from signalset."""
    signals = []

    commands = signalset.get("commands", [])

    if isinstance(commands, list):
        for cmd_data in commands:
            if not isinstance(cmd_data, dict):
                continue

            cmd_signals = cmd_data.get("signals", [])
            cmd_info = cmd_data.get("cmd", {})

            # Get PID
            pid = None
            if isinstance(cmd_info, dict):
                for service, pid_val in cmd_info.items():
                    pid = f"{service}:{pid_val}"
                    break

            for sig in cmd_signals:
                if not isinstance(sig, dict):
                    continue

                fmt = sig.get("fmt", {})
                unit = fmt.get("unit", "") if isinstance(fmt, dict) else ""

                signal = {
                    "id": sig.get("id", ""),
                    "name": sig.get("name", sig.get("id", "")),
                    "path": sig.get("path", ""),
                    "pid": pid,
                    "unit": unit,
                    "min": fmt.get("min") if isinstance(fmt, dict) else None,
                    "max": fmt.get("max") if isinstance(fmt, dict) else None,
                }
                signals.append(signal)

    elif isinstance(commands, dict):
        for cmd_id, cmd_data in commands.items():
            if not isinstance(cmd_data, dict):
                continue

            signal = {
                "id": cmd_id,
                "name": cmd_data.get("name", cmd_id),
                "unit": cmd_data.get("unit", ""),
                "min": cmd_data.get("min"),
                "max": cmd_data.get("max"),
            }
            signals.append(signal)

    return signals


def extract_dtcs(signalset: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract DTC codes from signalset."""
    dtcs = []

    # Try top-level dtcs
    dtc_data = signalset.get("dtcs", {})

    # Also check inside commands
    if not dtc_data:
        commands = signalset.get("commands", [])
        if isinstance(commands, list):
            for cmd in commands:
                if isinstance(cmd, dict):
                    cmd_dtcs = cmd.get("dtcs", {})
                    if isinstance(cmd_dtcs, dict):
                        dtc_data.update(cmd_dtcs)

    if isinstance(dtc_data, dict):
        for code, desc in dtc_data.items():
            if not re.match(r'^[PCBU][0-9A-F]{4}$', code, re.IGNORECASE):
                continue

            description = desc if isinstance(desc, str) else desc.get("description", "") if isinstance(desc, dict) else ""

            dtcs.append({
                "code": code.upper(),
                "description": description,
                "is_generic": code[1] == "0",
            })

    return dtcs


def parse_signalset(repo_name: str, data: Dict[str, Any]) -> ParsedVehicle:
    """Parse a signalset into structured vehicle data."""
    make, model = parse_repo_name(repo_name)

    signals = extract_signals(data)
    dtcs = extract_dtcs(data)

    commands = data.get("commands", [])
    raw_count = len(commands) if isinstance(commands, list) else len(commands) if isinstance(commands, dict) else 0

    return ParsedVehicle(
        repo_name=repo_name,
        make=make,
        model=model,
        signals=signals,
        dtcs=dtcs,
        raw_commands_count=raw_count,
    )


async def download_all(
    repos: List[RepoInfo],
    concurrency: int = DEFAULT_CONCURRENCY,
    force: bool = False,
) -> DownloadState:
    """
    Download signalsets from all repositories.

    Args:
        repos: List of repositories to download
        concurrency: Number of parallel downloads
        force: Force re-download even if already downloaded

    Returns:
        Updated download state
    """
    state = load_download_state() if not force else DownloadState()

    # Filter repos to download
    to_download = []
    for repo in repos:
        if repo.name in state.downloaded and not force:
            continue
        # Retry failed ones
        to_download.append(repo)

    if not to_download:
        logger.info("All repositories already downloaded!")
        return state

    logger.info(f"Downloading {len(to_download)} repositories ({len(state.downloaded)} already done)")

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(concurrency)

    # Ensure directories exist
    SIGNALSETS_DIR.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        tasks = [
            download_signalset(client, repo.name, semaphore)
            for repo in to_download
        ]

        # Process with progress bar
        results = await tqdm_asyncio.gather(*tasks, desc="Downloading signalsets")

        # Process results
        for repo_name, data, error in results:
            if data:
                await save_signalset(repo_name, data)
                state.downloaded.add(repo_name)
                state.failed.discard(repo_name)
            elif error == "No signalset found":
                state.skipped.add(repo_name)
                state.failed.discard(repo_name)
            else:
                state.failed.add(repo_name)
                logger.warning(f"{repo_name}: {error}")

        # Save state
        save_download_state(state)

    return state


def parse_all_downloaded() -> Dict[str, Any]:
    """Parse all downloaded signalsets."""
    logger.info("Parsing downloaded signalsets...")

    vehicles = {}
    total_signals = 0
    total_dtcs = 0
    errors = []

    signalset_files = list(SIGNALSETS_DIR.glob("*.json"))

    for file_path in tqdm(signalset_files, desc="Parsing files"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            repo_name = file_path.stem
            vehicle = parse_signalset(repo_name, data)

            vehicles[repo_name] = vehicle.to_dict()
            total_signals += len(vehicle.signals)
            total_dtcs += len(vehicle.dtcs)

        except Exception as e:
            errors.append({"file": str(file_path), "error": str(e)})

    result = {
        "vehicles": vehicles,
        "metadata": {
            "parsed_at": datetime.now(timezone.utc).isoformat(),
            "vehicle_count": len(vehicles),
            "total_signals": total_signals,
            "total_dtcs": total_dtcs,
            "parse_errors": len(errors),
        },
        "errors": errors,
    }

    # Save parsed data
    with open(PARSED_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info(f"Parsed {len(vehicles)} vehicles with {total_signals} signals and {total_dtcs} DTCs")

    return result


def generate_summary(repos: List[RepoInfo], state: DownloadState) -> Dict[str, Any]:
    """Generate download summary report."""
    # Count by make
    makes = {}
    for repo in repos:
        make = repo.make or "Unknown"
        makes[make] = makes.get(make, 0) + 1

    # Parse stats from downloaded files
    parsed_stats = {"vehicles": 0, "signals": 0, "dtcs": 0}
    if PARSED_DATA_FILE.exists():
        with open(PARSED_DATA_FILE, "r", encoding="utf-8") as f:
            parsed = json.load(f)
            parsed_stats = {
                "vehicles": parsed.get("metadata", {}).get("vehicle_count", 0),
                "signals": parsed.get("metadata", {}).get("total_signals", 0),
                "dtcs": parsed.get("metadata", {}).get("total_dtcs", 0),
            }

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_repos": len(repos),
        "downloaded": len(state.downloaded),
        "failed": len(state.failed),
        "skipped": len(state.skipped),
        "success_rate": f"{len(state.downloaded) / len(repos) * 100:.1f}%" if repos else "0%",
        "parsed": parsed_stats,
        "makes_distribution": dict(sorted(makes.items(), key=lambda x: -x[1])),
        "failed_repos": list(state.failed),
    }

    # Save summary
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    """Print formatted summary."""
    print("\n" + "=" * 70)
    print("OBDB DOWNLOAD SUMMARY")
    print("=" * 70)
    print(f"Generated: {summary['generated_at']}")
    print("-" * 70)
    print(f"Total repositories:    {summary['total_repos']}")
    print(f"Successfully downloaded: {summary['downloaded']}")
    print(f"Failed:                {summary['failed']}")
    print(f"Skipped (no signalset): {summary['skipped']}")
    print(f"Success rate:          {summary['success_rate']}")
    print("-" * 70)
    print("PARSED DATA:")
    print(f"  Vehicles:  {summary['parsed']['vehicles']}")
    print(f"  Signals:   {summary['parsed']['signals']}")
    print(f"  DTCs:      {summary['parsed']['dtcs']}")
    print("-" * 70)
    print("TOP 15 MAKES:")
    makes = summary.get("makes_distribution", {})
    for i, (make, count) in enumerate(list(makes.items())[:15]):
        print(f"  {make:25} {count:4} repos")
    if len(makes) > 15:
        print(f"  ... and {len(makes) - 15} more makes")
    print("=" * 70)

    if summary["failed"]:
        print("\nFAILED REPOS:")
        for repo in summary["failed_repos"][:10]:
            print(f"  - {repo}")
        if len(summary["failed_repos"]) > 10:
            print(f"  ... and {len(summary['failed_repos']) - 10} more")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download ALL OBDb repository signalsets"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all repositories without downloading",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download all signalsets (default action)",
    )
    parser.add_argument(
        "--parse",
        action="store_true",
        help="Parse downloaded signalsets",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show statistics only",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if already downloaded",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Number of parallel downloads (default: {DEFAULT_CONCURRENCY})",
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

    # Ensure directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SIGNALSETS_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load or fetch repos
    repos = []
    if REPOS_FILE.exists() and not args.force:
        logger.info("Loading cached repository list...")
        with open(REPOS_FILE, "r", encoding="utf-8") as f:
            repos_data = json.load(f)
            repos = [RepoInfo(**r) for r in repos_data]

    if not repos or args.list:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            repos = await fetch_all_repos(client)

        # Save repos list
        with open(REPOS_FILE, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in repos], f, indent=2)

    # Load download state
    state = load_download_state()

    # List only
    if args.list:
        print(f"\nFound {len(repos)} repositories:")
        for repo in repos[:50]:
            print(f"  {repo.make:20} {repo.model}")
        if len(repos) > 50:
            print(f"  ... and {len(repos) - 50} more")
        return

    # Stats only
    if args.stats:
        summary = generate_summary(repos, state)
        print_summary(summary)
        return

    # Download (default action)
    if args.download or not (args.parse or args.stats or args.list):
        state = await download_all(repos, concurrency=args.concurrency, force=args.force)

    # Parse
    if args.parse or args.download or not (args.stats or args.list):
        parse_all_downloaded()

    # Generate and show summary
    summary = generate_summary(repos, state)
    print_summary(summary)


if __name__ == "__main__":
    asyncio.run(main())
