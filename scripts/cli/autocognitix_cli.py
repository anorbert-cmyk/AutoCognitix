#!/usr/bin/env python3
"""
AutoCognitix CLI - Command-line interface for vehicle diagnostics.

A comprehensive CLI tool for the AutoCognitix vehicle diagnostic platform.
Supports DTC code lookup, vehicle diagnosis, VIN decoding, and translation tools.

Features:
- Colored output with rich library
- Progress bars for long operations
- Table formatting for results
- JSON output option (--json flag)
- Verbose mode (-v, -vv)
- Configuration file support (~/.autocognitix.yaml)

Usage:
    autocognitix diagnose P0171 P0174 --symptoms "Motor nehezen indul hidegben"
    autocognitix dtc search "oxygen sensor"
    autocognitix dtc info P0420
    autocognitix dtc related P0171
    autocognitix vehicle decode 1HGCM82633A123456
    autocognitix translate "Mass Air Flow Sensor"
    autocognitix stats

Author: AutoCognitix Team
"""

import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
import httpx
import yaml
from rich import box
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.tree import Tree

# Project paths
SCRIPT_DIR = Path(__file__).parent
SCRIPTS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
DTC_DATA_DIR = DATA_DIR / "dtc_codes"
CONFIG_FILE = Path.home() / ".autocognitix.yaml"

# DTC data files (priority order)
DTC_DATA_FILES = [
    DTC_DATA_DIR / "all_codes_merged.json",
    DTC_DATA_DIR / "generic_codes.json",
    DTC_DATA_DIR / "mytrile_codes.json",
    DTC_DATA_DIR / "klavkarr_codes.json",
]

TRANSLATION_CACHE_FILE = DTC_DATA_DIR / "translation_cache.json"

# DTC code format validation
DTC_PATTERN = re.compile(r"^[PBCU]\d{4}$", re.IGNORECASE)

# Category mappings
CATEGORY_MAP = {
    "P": "powertrain",
    "B": "body",
    "C": "chassis",
    "U": "network",
}

CATEGORY_NAMES = {
    "powertrain": {"en": "Powertrain", "hu": "Hajtaslancz"},
    "body": {"en": "Body", "hu": "Karosszeria"},
    "chassis": {"en": "Chassis", "hu": "Alvaaz"},
    "network": {"en": "Network", "hu": "Halozat/Kommunikacio"},
}

SEVERITY_COLORS = {
    "critical": "red",
    "high": "orange1",
    "medium": "yellow",
    "low": "green",
}

SEVERITY_NAMES = {
    "critical": {"en": "Critical", "hu": "Kritikus"},
    "high": {"en": "High", "hu": "Magas"},
    "medium": {"en": "Medium", "hu": "Kozepes"},
    "low": {"en": "Low", "hu": "Alacsony"},
}

# Console and logging setup
console = Console()


class Config:
    """Configuration manager with YAML file support."""

    DEFAULT_CONFIG = {
        "api_url": "http://localhost:8000/api/v1",
        "language": "hu",
        "default_limit": 20,
        "verbose": 0,
        "use_api": False,  # Use local data by default
    }

    def __init__(self):
        self.config_path = CONFIG_FILE
        self._config: Dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        """Load configuration from YAML file."""
        self._config = self.DEFAULT_CONFIG.copy()

        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    file_config = yaml.safe_load(f) or {}
                    self._config.update(file_config)
            except Exception as e:
                console.print(f"[yellow]Warning: Could not load config: {e}[/yellow]")

    def save(self) -> None:
        """Save configuration to YAML file."""
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.dump(self._config, f, default_flow_style=False)
        except Exception as e:
            console.print(f"[red]Error saving config: {e}[/red]")

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value."""
        self._config[key] = value

    @property
    def api_url(self) -> str:
        return self._config.get("api_url", self.DEFAULT_CONFIG["api_url"])

    @property
    def language(self) -> str:
        return self._config.get("language", "hu")


# Global config instance
config = Config()


class DTCDatabase:
    """Local DTC code database handler."""

    _instance: Optional["DTCDatabase"] = None
    _data: Optional[Dict[str, Any]] = None
    _code_index: Dict[str, Dict] = {}

    def __new__(cls) -> "DTCDatabase":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._data is None:
            self._load_data()

    def _load_data(self) -> None:
        """Load DTC data from JSON files."""
        for data_file in DTC_DATA_FILES:
            if data_file.exists():
                try:
                    with open(data_file, "r", encoding="utf-8") as f:
                        self._data = json.load(f)
                    break
                except json.JSONDecodeError as e:
                    console.print(f"[yellow]Warning: Could not parse {data_file}: {e}[/yellow]")

        if self._data is None:
            self._data = {"metadata": {}, "codes": []}

        # Build index by code
        self._code_index = {}
        for code_entry in self._data.get("codes", []):
            code = code_entry.get("code", "").upper()
            if code:
                self._code_index[code] = code_entry

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._data.get("metadata", {}) if self._data else {}

    @property
    def total_codes(self) -> int:
        return len(self._code_index)

    @property
    def all_codes(self) -> List[Dict[str, Any]]:
        return list(self._code_index.values())

    def get_code(self, code: str) -> Optional[Dict[str, Any]]:
        """Get a single DTC code by its identifier."""
        return self._code_index.get(code.upper())

    def search_codes(
        self,
        query: str,
        category: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Search codes by query string."""
        query_lower = query.lower()
        results = []

        for code, entry in self._code_index.items():
            if category and entry.get("category") != category:
                continue
            if severity and entry.get("severity") != severity:
                continue

            desc_en = entry.get("description_en") or ""
            desc_hu = entry.get("description_hu") or ""
            system = entry.get("system") or ""

            if (
                query_lower in code.lower()
                or query_lower in desc_en.lower()
                or query_lower in desc_hu.lower()
                or query_lower in system.lower()
            ):
                results.append(entry)

            if len(results) >= limit:
                break

        return results

    def get_related_codes(self, code: str) -> List[Dict[str, Any]]:
        """Get related DTC codes."""
        entry = self.get_code(code)
        if not entry:
            return []

        related = []
        for related_code in entry.get("related_codes", []):
            related_entry = self.get_code(related_code)
            if related_entry:
                related.append(related_entry)

        return related

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        categories = {}
        severities = {}
        systems = {}
        has_translation = 0
        has_symptoms = 0
        sources = {}

        for entry in self._code_index.values():
            cat = entry.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

            sev = entry.get("severity", "unknown")
            severities[sev] = severities.get(sev, 0) + 1

            sys_name = entry.get("system", "N/A")
            systems[sys_name] = systems.get(sys_name, 0) + 1

            if entry.get("description_hu"):
                has_translation += 1

            if entry.get("symptoms"):
                has_symptoms += 1

            for src in entry.get("sources", []):
                sources[src] = sources.get(src, 0) + 1

        return {
            "total_codes": self.total_codes,
            "categories": categories,
            "severities": severities,
            "systems": dict(sorted(systems.items(), key=lambda x: -x[1])[:20]),
            "translated": has_translation,
            "translation_percentage": round(has_translation / max(self.total_codes, 1) * 100, 1),
            "with_symptoms": has_symptoms,
            "sources": sources,
        }


class APIClient:
    """HTTP client for AutoCognitix API."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(30.0),
                headers={"Accept": "application/json"},
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def search_dtc(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Search DTC codes via API."""
        client = await self._get_client()
        params = {"q": query, "limit": limit}
        if category:
            params["category"] = category

        response = await client.get("/dtc/search", params=params)
        response.raise_for_status()
        return response.json()

    async def get_dtc_info(self, code: str) -> Optional[Dict[str, Any]]:
        """Get detailed DTC info via API."""
        client = await self._get_client()
        response = await client.get(f"/dtc/{code}")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    async def get_related_codes(self, code: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get related DTC codes via API."""
        client = await self._get_client()
        response = await client.get(f"/dtc/{code}/related", params={"limit": limit})
        if response.status_code == 404:
            return []
        response.raise_for_status()
        return response.json()

    async def decode_vin(self, vin: str) -> Optional[Dict[str, Any]]:
        """Decode VIN via NHTSA service."""
        client = await self._get_client()
        response = await client.get(f"/vehicles/decode-vin/{vin}")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    async def diagnose(
        self,
        dtc_codes: List[str],
        symptoms: str,
        make: Optional[str] = None,
        model: Optional[str] = None,
        year: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run diagnosis via API."""
        client = await self._get_client()
        data = {
            "dtc_codes": dtc_codes,
            "symptoms": symptoms,
            "vehicle_make": make or "Unknown",
            "vehicle_model": model or "Unknown",
            "vehicle_year": year or 2020,
        }
        response = await client.post("/diagnosis/analyze", json=data)
        response.raise_for_status()
        return response.json()


def validate_dtc_code(code: str) -> bool:
    """Validate DTC code format."""
    return bool(DTC_PATTERN.match(code.upper()))


def setup_logging(verbosity: int) -> None:
    """Configure logging based on verbosity level."""
    if verbosity == 0:
        level = logging.WARNING
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False, show_time=False)],
    )


def output_json(data: Any) -> None:
    """Output data as formatted JSON."""
    console.print(json.dumps(data, ensure_ascii=False, indent=2, default=str))


# =============================================================================
# Click CLI Application
# =============================================================================

class Context:
    """CLI context for passing config between commands."""

    def __init__(self):
        self.verbose: int = 0
        self.json_output: bool = False
        self.config: Config = config


pass_context = click.make_pass_decorator(Context, ensure=True)


@click.group()
@click.option("-v", "--verbose", count=True, help="Increase verbosity (-v, -vv)")
@click.option("--json", "json_output", is_flag=True, help="Output results as JSON")
@click.option("--config", "config_file", type=click.Path(), help="Config file path")
@click.version_option(version="1.0.0", prog_name="autocognitix")
@pass_context
def cli(ctx: Context, verbose: int, json_output: bool, config_file: Optional[str]):
    """
    AutoCognitix CLI - Vehicle diagnostic command-line tool.

    Provides DTC code lookup, vehicle diagnosis, VIN decoding,
    and translation utilities for the AutoCognitix platform.
    """
    ctx.verbose = verbose
    ctx.json_output = json_output
    setup_logging(verbose)

    if config_file:
        ctx.config.config_path = Path(config_file)
        ctx.config.load()


# =============================================================================
# Diagnose Command
# =============================================================================

@cli.command("diagnose")
@click.argument("dtc_codes", nargs=-1, required=True)
@click.option("--symptoms", "-s", default="", help="Symptom description (Hungarian)")
@click.option("--make", "-m", help="Vehicle make")
@click.option("--model", help="Vehicle model")
@click.option("--year", "-y", type=int, help="Vehicle year")
@click.option("--use-api", is_flag=True, help="Use backend API instead of local data")
@pass_context
def diagnose_cmd(
    ctx: Context,
    dtc_codes: Tuple[str, ...],
    symptoms: str,
    make: Optional[str],
    model: Optional[str],
    year: Optional[int],
    use_api: bool,
):
    """
    Run vehicle diagnosis with DTC codes and symptoms.

    Examples:

        autocognitix diagnose P0171 P0174 --symptoms "Motor nehezen indul"

        autocognitix diagnose P0420 -s "Fogyasztas novekedett" --make VW --model Golf
    """
    # Validate DTC codes
    valid_codes = []
    for code in dtc_codes:
        code_upper = code.upper()
        if validate_dtc_code(code_upper):
            valid_codes.append(code_upper)
        else:
            console.print(f"[yellow]Warning: Invalid DTC format skipped: {code}[/yellow]")

    if not valid_codes:
        console.print("[red]Error: No valid DTC codes provided[/red]")
        raise SystemExit(1)

    if use_api or ctx.config.get("use_api"):
        # Use API for diagnosis
        asyncio.run(_diagnose_via_api(ctx, valid_codes, symptoms, make, model, year))
    else:
        # Use local data
        _diagnose_local(ctx, valid_codes, symptoms, make, model, year)


def _diagnose_local(
    ctx: Context,
    codes: List[str],
    symptoms: str,
    make: Optional[str],
    model: Optional[str],
    year: Optional[int],
):
    """Run diagnosis using local DTC database."""
    db = DTCDatabase()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Analyzing DTC codes...", total=None)

        found_codes = []
        not_found = []

        for code in codes:
            entry = db.get_code(code)
            if entry:
                found_codes.append(entry)
            else:
                not_found.append(code)

    if ctx.json_output:
        result = {
            "diagnosis": {
                "dtc_codes": codes,
                "symptoms": symptoms,
                "vehicle": {"make": make, "model": model, "year": year},
                "found_codes": found_codes,
                "not_found": not_found,
                "timestamp": datetime.now().isoformat(),
            }
        }
        output_json(result)
        return

    # Pretty output
    console.print()
    title = "[bold blue]DIAGNOSIS RESULTS[/bold blue]"
    if make or model or year:
        title += f" - {make or ''} {model or ''} {year or ''}"
    console.print(Panel(title, box=box.DOUBLE))

    if symptoms:
        console.print(f"\n[bold]Symptoms:[/bold] {symptoms}")

    if not_found:
        console.print(f"\n[yellow]Codes not in database: {', '.join(not_found)}[/yellow]")

    if found_codes:
        console.print(f"\n[bold]DTC Code Analysis ({len(found_codes)} codes):[/bold]")

        for severity_level in ["critical", "high", "medium", "low"]:
            codes_at_level = [c for c in found_codes if c.get("severity") == severity_level]
            if codes_at_level:
                color = SEVERITY_COLORS.get(severity_level, "white")
                lang = ctx.config.language
                sev_name = SEVERITY_NAMES.get(severity_level, {}).get(lang, severity_level)
                console.print(f"\n[bold {color}]{sev_name.upper()}:[/bold {color}]")

                for entry in codes_at_level:
                    desc = entry.get(f"description_{lang}") or entry.get("description_en", "")
                    console.print(f"  [{color}]{entry.get('code')}[/{color}] - {desc}")

                    if entry.get("possible_causes"):
                        causes = "; ".join(entry.get("possible_causes", [])[:3])
                        console.print(f"    [dim]Causes: {causes}[/dim]")

        # Recommendations
        console.print("\n[bold]Recommendations:[/bold]")
        console.print("  1. Address critical and high severity codes first")
        console.print("  2. Follow diagnostic steps in order")
        console.print("  3. Check related codes for root cause analysis")

        if any(c.get("severity") == "critical" for c in found_codes):
            console.print(
                "\n[bold red]WARNING: Critical fault code detected! "
                "Immediate professional inspection recommended.[/bold red]"
            )


async def _diagnose_via_api(
    ctx: Context,
    codes: List[str],
    symptoms: str,
    make: Optional[str],
    model: Optional[str],
    year: Optional[int],
):
    """Run diagnosis using backend API."""
    client = APIClient(ctx.config.api_url)

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Running diagnosis via API...", total=None)
            result = await client.diagnose(codes, symptoms, make, model, year)

        if ctx.json_output:
            output_json(result)
        else:
            _display_api_diagnosis_result(ctx, result)

    except httpx.HTTPError as e:
        console.print(f"[red]API Error: {e}[/red]")
        raise SystemExit(1)
    finally:
        await client.close()


def _display_api_diagnosis_result(ctx: Context, result: Dict[str, Any]):
    """Display diagnosis result from API."""
    console.print()
    console.print(Panel("[bold blue]DIAGNOSIS RESULTS[/bold blue]", box=box.DOUBLE))

    console.print(f"\n[bold]Vehicle:[/bold] {result.get('vehicle_make', '')} "
                  f"{result.get('vehicle_model', '')} {result.get('vehicle_year', '')}")
    console.print(f"[bold]DTC Codes:[/bold] {', '.join(result.get('dtc_codes', []))}")
    console.print(f"[bold]Confidence:[/bold] {result.get('confidence_score', 0):.1%}")

    # Probable causes
    causes = result.get("probable_causes", [])
    if causes:
        console.print("\n[bold]Probable Causes:[/bold]")
        for cause in causes[:5]:
            confidence = cause.get("confidence", 0)
            console.print(f"  - {cause.get('title', '')} ({confidence:.0%})")
            if cause.get("description"):
                console.print(f"    [dim]{cause['description'][:100]}...[/dim]")

    # Recommendations
    repairs = result.get("recommended_repairs", [])
    if repairs:
        console.print("\n[bold]Recommended Repairs:[/bold]")
        for repair in repairs[:5]:
            console.print(f"  - {repair.get('title', '')}")


# =============================================================================
# DTC Commands Group
# =============================================================================

@cli.group("dtc")
def dtc_group():
    """DTC code lookup and search commands."""
    pass


@dtc_group.command("search")
@click.argument("query")
@click.option("--category", "-c", type=click.Choice(["powertrain", "body", "chassis", "network"]))
@click.option("--limit", "-l", default=20, help="Maximum results")
@click.option("--use-api", is_flag=True, help="Use backend API")
@pass_context
def dtc_search(ctx: Context, query: str, category: Optional[str], limit: int, use_api: bool):
    """
    Search DTC codes by description or code.

    Examples:

        autocognitix dtc search "oxygen sensor"

        autocognitix dtc search P01 --category powertrain
    """
    if use_api or ctx.config.get("use_api"):
        asyncio.run(_dtc_search_api(ctx, query, category, limit))
    else:
        _dtc_search_local(ctx, query, category, limit)


def _dtc_search_local(ctx: Context, query: str, category: Optional[str], limit: int):
    """Search DTC codes in local database."""
    db = DTCDatabase()
    results = db.search_codes(query, category=category, limit=limit)

    if not results:
        console.print(f"[yellow]No results for: {query}[/yellow]")
        return

    if ctx.json_output:
        output_json({"query": query, "count": len(results), "codes": results})
        return

    table = Table(
        title=f"Search Results: '{query}' ({len(results)} found)",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )

    table.add_column("Code", style="bold", width=8)
    table.add_column("Description", width=45)
    table.add_column("Category", width=12)
    table.add_column("Severity", width=10)
    table.add_column("System", width=18)

    lang = ctx.config.language
    for entry in results:
        severity = entry.get("severity", "medium")
        severity_color = SEVERITY_COLORS.get(severity, "white")
        desc = entry.get(f"description_{lang}") or entry.get("description_en", "N/A")
        cat_name = CATEGORY_NAMES.get(entry.get("category", ""), {}).get(lang, entry.get("category", ""))

        table.add_row(
            entry.get("code", "???"),
            desc[:45] if desc else "N/A",
            cat_name[:12] if cat_name else "",
            f"[{severity_color}]{SEVERITY_NAMES.get(severity, {}).get(lang, severity)}[/{severity_color}]",
            (entry.get("system") or "N/A")[:18],
        )

    console.print(table)


async def _dtc_search_api(ctx: Context, query: str, category: Optional[str], limit: int):
    """Search DTC codes via API."""
    client = APIClient(ctx.config.api_url)

    try:
        results = await client.search_dtc(query, category, limit)

        if ctx.json_output:
            output_json({"query": query, "count": len(results), "codes": results})
            return

        if not results:
            console.print(f"[yellow]No results for: {query}[/yellow]")
            return

        table = Table(
            title=f"Search Results: '{query}' ({len(results)} found)",
            box=box.ROUNDED,
        )
        table.add_column("Code", style="bold", width=8)
        table.add_column("Description", width=50)
        table.add_column("Category", width=12)

        for entry in results:
            desc = entry.get("description_hu") or entry.get("description_en", "")
            table.add_row(entry.get("code", ""), desc[:50], entry.get("category", ""))

        console.print(table)

    except httpx.HTTPError as e:
        console.print(f"[red]API Error: {e}[/red]")
    finally:
        await client.close()


@dtc_group.command("info")
@click.argument("code")
@click.option("--use-api", is_flag=True, help="Use backend API")
@pass_context
def dtc_info(ctx: Context, code: str, use_api: bool):
    """
    Get detailed information about a DTC code.

    Examples:

        autocognitix dtc info P0420

        autocognitix dtc info P0171 --json
    """
    code = code.upper().strip()

    if not validate_dtc_code(code):
        console.print(f"[red]Invalid DTC format: {code}[/red]")
        console.print("[dim]Valid format: [P/B/C/U] + 4 digits (e.g., P0171)[/dim]")
        raise SystemExit(1)

    if use_api or ctx.config.get("use_api"):
        asyncio.run(_dtc_info_api(ctx, code))
    else:
        _dtc_info_local(ctx, code)


def _dtc_info_local(ctx: Context, code: str):
    """Get DTC info from local database."""
    db = DTCDatabase()
    entry = db.get_code(code)

    if not entry:
        console.print(f"[yellow]DTC code not found: {code}[/yellow]")
        raise SystemExit(1)

    if ctx.json_output:
        output_json(entry)
        return

    lang = ctx.config.language
    severity = entry.get("severity", "medium")
    severity_color = SEVERITY_COLORS.get(severity, "white")

    desc = entry.get(f"description_{lang}") or entry.get("description_en", "")
    desc_alt = entry.get("description_en" if lang == "hu" else "description_hu", "")

    content_lines = []
    if desc:
        content_lines.append(f"[bold]{desc}[/bold]")
    if desc_alt and desc_alt != desc:
        content_lines.append(f"[dim]{desc_alt}[/dim]")

    content_lines.append("")
    cat_name = CATEGORY_NAMES.get(entry.get("category", ""), {}).get(lang, entry.get("category", ""))
    content_lines.append(f"[bold]Category:[/bold] {cat_name}")
    content_lines.append(f"[bold]System:[/bold] {entry.get('system', 'N/A')}")
    sev_name = SEVERITY_NAMES.get(severity, {}).get(lang, severity)
    content_lines.append(f"[bold]Severity:[/bold] [{severity_color}]{sev_name}[/{severity_color}]")
    content_lines.append(f"[bold]Generic:[/bold] {'Yes' if entry.get('is_generic') else 'No'}")

    symptoms = entry.get("symptoms", [])
    if symptoms:
        content_lines.append("")
        content_lines.append("[bold]Symptoms:[/bold]")
        for symptom in symptoms[:6]:
            content_lines.append(f"  - {symptom}")

    causes = entry.get("possible_causes", [])
    if causes:
        content_lines.append("")
        content_lines.append("[bold]Possible Causes:[/bold]")
        for cause in causes[:6]:
            content_lines.append(f"  - {cause}")

    steps = entry.get("diagnostic_steps", [])
    if steps:
        content_lines.append("")
        content_lines.append("[bold]Diagnostic Steps:[/bold]")
        for i, step in enumerate(steps[:6], 1):
            content_lines.append(f"  {i}. {step}")

    related = entry.get("related_codes", [])
    if related:
        content_lines.append("")
        content_lines.append(f"[bold]Related Codes:[/bold] {', '.join(related[:10])}")

    sources = entry.get("sources", [])
    if sources:
        content_lines.append("")
        content_lines.append(f"[dim]Sources: {', '.join(sources)}[/dim]")

    panel = Panel(
        "\n".join(content_lines),
        title=f"[bold]{code}[/bold]",
        border_style=severity_color,
        box=box.ROUNDED,
    )
    console.print(panel)


async def _dtc_info_api(ctx: Context, code: str):
    """Get DTC info via API."""
    client = APIClient(ctx.config.api_url)

    try:
        result = await client.get_dtc_info(code)

        if result is None:
            console.print(f"[yellow]DTC code not found: {code}[/yellow]")
            raise SystemExit(1)

        if ctx.json_output:
            output_json(result)
        else:
            # Display similar to local format
            _display_dtc_detail(ctx, result)

    except httpx.HTTPError as e:
        console.print(f"[red]API Error: {e}[/red]")
        raise SystemExit(1)
    finally:
        await client.close()


def _display_dtc_detail(ctx: Context, entry: Dict[str, Any]):
    """Display DTC detail from API response."""
    code = entry.get("code", "???")
    severity = entry.get("severity", "medium")
    severity_color = SEVERITY_COLORS.get(severity, "white")

    lang = ctx.config.language
    desc = entry.get(f"description_{lang}") or entry.get("description_en", "")

    content_lines = [f"[bold]{desc}[/bold]", ""]
    content_lines.append(f"[bold]Category:[/bold] {entry.get('category', 'N/A')}")
    content_lines.append(f"[bold]System:[/bold] {entry.get('system', 'N/A')}")
    content_lines.append(f"[bold]Severity:[/bold] [{severity_color}]{severity}[/{severity_color}]")

    panel = Panel("\n".join(content_lines), title=f"[bold]{code}[/bold]", border_style=severity_color)
    console.print(panel)


@dtc_group.command("related")
@click.argument("code")
@click.option("--limit", "-l", default=10, help="Maximum results")
@click.option("--use-api", is_flag=True, help="Use backend API")
@pass_context
def dtc_related(ctx: Context, code: str, limit: int, use_api: bool):
    """
    Find DTC codes related to a specific code.

    Examples:

        autocognitix dtc related P0171

        autocognitix dtc related P0300 --limit 5
    """
    code = code.upper().strip()

    if not validate_dtc_code(code):
        console.print(f"[red]Invalid DTC format: {code}[/red]")
        raise SystemExit(1)

    if use_api or ctx.config.get("use_api"):
        asyncio.run(_dtc_related_api(ctx, code, limit))
    else:
        _dtc_related_local(ctx, code, limit)


def _dtc_related_local(ctx: Context, code: str, limit: int):
    """Get related codes from local database."""
    db = DTCDatabase()
    entry = db.get_code(code)

    if not entry:
        console.print(f"[yellow]DTC code not found: {code}[/yellow]")
        raise SystemExit(1)

    related = db.get_related_codes(code)[:limit]
    related_strs = entry.get("related_codes", [])

    if not related and not related_strs:
        console.print(f"[yellow]No related codes found for: {code}[/yellow]")
        return

    if ctx.json_output:
        output_json({"code": code, "related_codes": related_strs, "related_details": related})
        return

    console.print(f"\n[bold]Related codes for {code}:[/bold]\n")

    if related:
        lang = ctx.config.language
        table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
        table.add_column("Code", style="bold", width=8)
        table.add_column("Description", width=50)
        table.add_column("Severity", width=10)

        for rel_entry in related:
            severity = rel_entry.get("severity", "medium")
            severity_color = SEVERITY_COLORS.get(severity, "white")
            desc = rel_entry.get(f"description_{lang}") or rel_entry.get("description_en", "N/A")

            table.add_row(
                rel_entry.get("code", "???"),
                desc[:50] if desc else "N/A",
                f"[{severity_color}]{SEVERITY_NAMES.get(severity, {}).get(lang, severity)}[/{severity_color}]",
            )

        console.print(table)
    else:
        console.print(f"[dim]Related codes (not in database): {', '.join(related_strs)}[/dim]")


async def _dtc_related_api(ctx: Context, code: str, limit: int):
    """Get related codes via API."""
    client = APIClient(ctx.config.api_url)

    try:
        results = await client.get_related_codes(code, limit)

        if ctx.json_output:
            output_json({"code": code, "related": results})
            return

        if not results:
            console.print(f"[yellow]No related codes found for: {code}[/yellow]")
            return

        table = Table(title=f"Related codes for {code}", box=box.ROUNDED)
        table.add_column("Code", style="bold", width=8)
        table.add_column("Description", width=50)

        for entry in results:
            desc = entry.get("description_hu") or entry.get("description_en", "")
            table.add_row(entry.get("code", ""), desc[:50])

        console.print(table)

    except httpx.HTTPError as e:
        console.print(f"[red]API Error: {e}[/red]")
    finally:
        await client.close()


# =============================================================================
# Vehicle Commands Group
# =============================================================================

@cli.group("vehicle")
def vehicle_group():
    """Vehicle-related commands."""
    pass


@vehicle_group.command("decode")
@click.argument("vin")
@pass_context
def vehicle_decode(ctx: Context, vin: str):
    """
    Decode a VIN (Vehicle Identification Number).

    Examples:

        autocognitix vehicle decode 1HGCM82633A123456

        autocognitix vehicle decode WVWZZZ3CZWE123456 --json
    """
    vin = vin.strip().upper()

    if len(vin) != 17:
        console.print(f"[red]Invalid VIN length: {len(vin)} (expected 17)[/red]")
        raise SystemExit(1)

    asyncio.run(_decode_vin(ctx, vin))


async def _decode_vin(ctx: Context, vin: str):
    """Decode VIN using NHTSA API."""
    # Try direct NHTSA API call
    url = f"https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVinValues/{vin}?format=json"

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Decoding VIN...", total=None)

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
                results = data.get("Results", [{}])[0]
            except httpx.HTTPError as e:
                console.print(f"[red]Error decoding VIN: {e}[/red]")
                raise SystemExit(1)

    if ctx.json_output:
        output_json(results)
        return

    console.print(Panel(f"[bold]VIN: {vin}[/bold]", box=box.DOUBLE))

    table = Table(box=box.SIMPLE, show_header=False)
    table.add_column("Field", style="bold", width=25)
    table.add_column("Value", width=40)

    fields = [
        ("Make", "Make"),
        ("Model", "Model"),
        ("Year", "ModelYear"),
        ("Body Class", "BodyClass"),
        ("Vehicle Type", "VehicleType"),
        ("Plant Country", "PlantCountry"),
        ("Manufacturer", "Manufacturer"),
        ("Engine Cylinders", "EngineCylinders"),
        ("Engine Displacement (L)", "DisplacementL"),
        ("Fuel Type", "FuelTypePrimary"),
        ("Transmission", "TransmissionStyle"),
        ("Drive Type", "DriveType"),
    ]

    for label, key in fields:
        value = results.get(key, "")
        if value and value.strip():
            table.add_row(label, str(value))

    console.print(table)


# =============================================================================
# Translate Command
# =============================================================================

@cli.command("translate")
@click.argument("text")
@pass_context
def translate_cmd(ctx: Context, text: str):
    """
    Translate automotive term to Hungarian.

    Examples:

        autocognitix translate "Mass Air Flow Sensor"

        autocognitix translate "Catalytic Converter Efficiency"
    """
    # Check if it's a DTC code
    if validate_dtc_code(text):
        db = DTCDatabase()
        entry = db.get_code(text.upper())

        if entry and entry.get("description_hu"):
            if ctx.json_output:
                output_json({
                    "code": text.upper(),
                    "english": entry.get("description_en", ""),
                    "hungarian": entry.get("description_hu", ""),
                })
            else:
                console.print(f"[bold]{text.upper()}[/bold]")
                console.print(f"  EN: {entry.get('description_en', 'N/A')}")
                console.print(f"  HU: [green]{entry.get('description_hu')}[/green]")
            return

    # Check translation cache
    cache = {}
    if TRANSLATION_CACHE_FILE.exists():
        try:
            with open(TRANSLATION_CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            pass

    # Look for exact or partial match
    text_lower = text.lower()
    for key, value in cache.items():
        if text_lower in key.lower() or key.lower() in text_lower:
            if ctx.json_output:
                output_json({"english": key, "hungarian": value})
            else:
                console.print(f"  EN: {key}")
                console.print(f"  HU: [green]{value}[/green]")
            return

    # No translation found
    if ctx.json_output:
        output_json({"english": text, "hungarian": None, "message": "No translation found"})
    else:
        console.print(f"[yellow]No translation found for: {text}[/yellow]")
        console.print("[dim]Use the translation script to add new translations:[/dim]")
        console.print("[dim]  python scripts/translate_to_hungarian.py[/dim]")


# =============================================================================
# Stats Command
# =============================================================================

@cli.command("stats")
@pass_context
def stats_cmd(ctx: Context):
    """
    Show database statistics.

    Displays counts for DTC codes, categories, translations, etc.
    """
    db = DTCDatabase()
    stats = db.get_statistics()

    if ctx.json_output:
        output_json(stats)
        return

    console.print(Panel("[bold]AutoCognitix Database Statistics[/bold]", box=box.DOUBLE))

    # Overview table
    overview = Table(title="Overview", box=box.ROUNDED, show_header=False)
    overview.add_column("Metric", style="bold", width=30)
    overview.add_column("Value", justify="right", width=15)

    overview.add_row("Total DTC Codes", str(stats["total_codes"]))
    overview.add_row("Translated Codes", f"{stats['translated']} ({stats['translation_percentage']}%)")
    overview.add_row("Codes with Symptoms", str(stats["with_symptoms"]))

    console.print(overview)

    # Category breakdown
    cat_table = Table(title="Categories", box=box.SIMPLE)
    cat_table.add_column("Category", width=20)
    cat_table.add_column("Count", justify="right", width=10)

    for cat, count in sorted(stats["categories"].items(), key=lambda x: -x[1]):
        cat_name = CATEGORY_NAMES.get(cat, {}).get("en", cat)
        cat_table.add_row(cat_name, str(count))

    console.print(cat_table)

    # Severity breakdown
    sev_table = Table(title="Severity Levels", box=box.SIMPLE)
    sev_table.add_column("Severity", width=15)
    sev_table.add_column("Count", justify="right", width=10)

    for sev, count in sorted(stats["severities"].items(), key=lambda x: -x[1]):
        color = SEVERITY_COLORS.get(sev, "white")
        sev_name = SEVERITY_NAMES.get(sev, {}).get("en", sev)
        sev_table.add_row(f"[{color}]{sev_name}[/{color}]", str(count))

    console.print(sev_table)

    # Translation progress bar
    progress_pct = stats["translation_percentage"]
    bar_width = 40
    filled = int(bar_width * progress_pct / 100)
    bar = "[green]" + "=" * filled + "[/green]" + "[dim]" + "-" * (bar_width - filled) + "[/dim]"
    console.print(f"\nTranslation Progress: [{bar}] {progress_pct}%")


# =============================================================================
# Config Command
# =============================================================================

@cli.command("config")
@click.option("--show", is_flag=True, help="Show current configuration")
@click.option("--set", "set_value", nargs=2, multiple=True, help="Set config value (key value)")
@click.option("--reset", is_flag=True, help="Reset to default configuration")
@pass_context
def config_cmd(ctx: Context, show: bool, set_value: Tuple[Tuple[str, str], ...], reset: bool):
    """
    Manage CLI configuration.

    Examples:

        autocognitix config --show

        autocognitix config --set api_url http://localhost:8000/api/v1

        autocognitix config --set language en
    """
    if reset:
        ctx.config._config = ctx.config.DEFAULT_CONFIG.copy()
        ctx.config.save()
        console.print("[green]Configuration reset to defaults[/green]")
        return

    if set_value:
        for key, value in set_value:
            # Convert types
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            elif value.isdigit():
                value = int(value)

            ctx.config.set(key, value)

        ctx.config.save()
        console.print("[green]Configuration saved[/green]")

    if show or (not set_value and not reset):
        console.print(Panel("[bold]Current Configuration[/bold]", box=box.ROUNDED))
        console.print(f"Config file: {ctx.config.config_path}")
        console.print()

        for key, value in sorted(ctx.config._config.items()):
            console.print(f"  [bold]{key}:[/bold] {value}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
