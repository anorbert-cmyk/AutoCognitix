#!/usr/bin/env python3
"""
AutoCognitix Diagnostic Tool - Comprehensive CLI Interface
===========================================================

Interactive CLI for vehicle diagnostics with DTC code lookup,
vehicle database queries, statistics, and translation tools.

Features:
- DTC Code Lookup (lookup, search, related)
- Quick Diagnosis with DTC codes and symptoms
- Vehicle Selection (makes, models, VIN decode)
- Database Statistics
- Translation Tools

Usage:
    # DTC Lookup
    diagtool lookup P0171
    diagtool search "oxygen sensor"
    diagtool related P0171

    # Diagnosis
    diagtool diagnose --codes P0171,P0101 --symptoms "motor vibral"
    diagtool diagnose --interactive

    # Vehicles
    diagtool vehicles list
    diagtool vehicles models VW
    diagtool vin decode WVWZZZ3CZWE123456

    # Statistics
    diagtool stats
    diagtool stats dtc
    diagtool stats translations

    # Translation
    diagtool translate P0171
    diagtool translate-batch --limit 100

Author: AutoCognitix Team
"""

import asyncio
import json
import os
import re
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path for imports
SCRIPT_DIR = Path(__file__).parent
SCRIPTS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

try:
    import typer
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.prompt import Confirm, Prompt
    from rich.table import Table
    from rich.tree import Tree
    from rich.syntax import Syntax
except ImportError:
    print("Error: Missing dependencies. Install with: pip install typer[all] rich")
    sys.exit(1)


# =============================================================================
# Configuration & Constants
# =============================================================================

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
DTC_DATA_DIR = DATA_DIR / "dtc_codes"
VEHICLES_DIR = DATA_DIR / "vehicles"

# DTC data files (priority order)
DTC_DATA_FILES = [
    DTC_DATA_DIR / "all_codes_merged.json",
    DTC_DATA_DIR / "generic_codes.json",
    DTC_DATA_DIR / "mytrile_codes.json",
    DTC_DATA_DIR / "klavkarr_codes.json",
]

TRANSLATION_CACHE_FILE = DTC_DATA_DIR / "translation_cache.json"
VEHICLES_CACHE_FILE = VEHICLES_DIR / "obdb_cache" / "vehicles.json"

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

# =============================================================================
# Enums
# =============================================================================


class OutputFormat(str, Enum):
    """Output format options."""
    TABLE = "table"
    JSON = "json"
    CSV = "csv"


class Language(str, Enum):
    """Language options."""
    HU = "hu"
    EN = "en"


# =============================================================================
# Data Access Classes
# =============================================================================


class DTCDatabase:
    """DTC code database handler with caching."""

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
                with open(data_file, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
                break
        else:
            self._data = {"metadata": {}, "codes": []}

        # Build index by code
        self._code_index = {}
        for code_entry in self._data.get("codes", []):
            code = code_entry.get("code", "").upper()
            if code:
                self._code_index[code] = code_entry

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get database metadata."""
        return self._data.get("metadata", {}) if self._data else {}

    @property
    def total_codes(self) -> int:
        """Get total number of codes."""
        return len(self._code_index)

    @property
    def all_codes(self) -> List[Dict[str, Any]]:
        """Get all codes."""
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

    def search_by_symptoms(
        self,
        symptoms: str,
        limit: int = 20,
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """Search codes by symptom description. Returns list of (score, entry)."""
        symptom_words = set(
            word.lower()
            for word in re.findall(r"\w+", symptoms)
            if len(word) > 2
        )

        scored_results = []

        for code, entry in self._code_index.items():
            score = 0.0

            for symptom in entry.get("symptoms", []):
                symptom_lower = symptom.lower()
                for word in symptom_words:
                    if word in symptom_lower:
                        score += 2

            for cause in entry.get("possible_causes", []):
                cause_lower = cause.lower()
                for word in symptom_words:
                    if word in cause_lower:
                        score += 1

            desc_hu = (entry.get("description_hu") or "").lower()
            desc_en = (entry.get("description_en") or "").lower()
            for word in symptom_words:
                if word in desc_hu:
                    score += 1.5
                if word in desc_en:
                    score += 0.5

            if score > 0:
                scored_results.append((score, entry))

        scored_results.sort(key=lambda x: x[0], reverse=True)
        return scored_results[:limit]

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


class VehicleDatabase:
    """Vehicle database handler."""

    _instance: Optional["VehicleDatabase"] = None
    _data: Optional[Dict[str, Any]] = None

    def __new__(cls) -> "VehicleDatabase":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._data is None:
            self._load_data()

    def _load_data(self) -> None:
        """Load vehicle data."""
        self._data = {"vehicles": {}}

        if VEHICLES_CACHE_FILE.exists():
            with open(VEHICLES_CACHE_FILE, "r", encoding="utf-8") as f:
                self._data = json.load(f)

    def get_makes(self) -> List[str]:
        """Get unique vehicle makes."""
        makes = set()
        for vehicle in self._data.get("vehicles", {}).values():
            make = vehicle.get("make", "")
            if make:
                makes.add(make)
        return sorted(makes)

    def get_models(self, make: str) -> List[Dict[str, Any]]:
        """Get models for a specific make."""
        make_lower = make.lower()
        models = []

        for vehicle in self._data.get("vehicles", {}).values():
            vehicle_make = vehicle.get("make", "").lower()
            if make_lower in vehicle_make or vehicle_make in make_lower:
                models.append({
                    "id": vehicle.get("id", ""),
                    "make": vehicle.get("make", ""),
                    "model": vehicle.get("model", ""),
                    "dtc_count": vehicle.get("dtc_count", 0),
                    "signal_count": vehicle.get("signal_count", 0),
                })

        return models

    def get_vehicle(self, vehicle_id: str) -> Optional[Dict[str, Any]]:
        """Get vehicle by ID."""
        return self._data.get("vehicles", {}).get(vehicle_id)

    @property
    def total_vehicles(self) -> int:
        """Get total number of vehicles."""
        return len(self._data.get("vehicles", {}))


class TranslationCache:
    """Translation cache handler."""

    _instance: Optional["TranslationCache"] = None
    _cache: Optional[Dict[str, str]] = None

    def __new__(cls) -> "TranslationCache":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._cache is None:
            self._load_cache()

    def _load_cache(self) -> None:
        """Load translation cache."""
        self._cache = {}
        if TRANSLATION_CACHE_FILE.exists():
            with open(TRANSLATION_CACHE_FILE, "r", encoding="utf-8") as f:
                self._cache = json.load(f)

    def get(self, code: str) -> Optional[str]:
        """Get translation for a code."""
        return self._cache.get(code.upper())

    def get_statistics(self) -> Dict[str, Any]:
        """Get translation statistics."""
        dtc_db = DTCDatabase()

        total_codes = dtc_db.total_codes
        cached = len(self._cache)
        translated_in_db = sum(
            1 for entry in dtc_db.all_codes if entry.get("description_hu")
        )

        return {
            "total_codes": total_codes,
            "cached_translations": cached,
            "translated_in_database": translated_in_db,
            "translation_percentage": round(translated_in_db / max(total_codes, 1) * 100, 1),
            "pending": total_codes - translated_in_db,
        }


# =============================================================================
# CLI Application
# =============================================================================

app = typer.Typer(
    name="diagtool",
    help="AutoCognitix Diagnostic Tool - Vehicle diagnostics CLI",
    add_completion=True,
    rich_markup_mode="rich",
    no_args_is_help=True,
)

# Sub-applications for grouping
vehicles_app = typer.Typer(
    name="vehicles",
    help="Vehicle database commands",
    no_args_is_help=True,
)
stats_app = typer.Typer(
    name="stats",
    help="Database statistics commands",
)
vin_app = typer.Typer(
    name="vin",
    help="VIN operations",
    no_args_is_help=True,
)

app.add_typer(vehicles_app, name="vehicles")
app.add_typer(stats_app, name="stats")
app.add_typer(vin_app, name="vin")

console = Console()


def validate_dtc_code(code: str) -> bool:
    """Validate DTC code format."""
    return bool(DTC_PATTERN.match(code.upper()))


def get_language_text(data: Dict[str, str], lang: Language) -> str:
    """Get text in specified language."""
    return data.get(lang.value, data.get("en", ""))


# =============================================================================
# DTC Lookup Commands
# =============================================================================


@app.command("lookup")
def lookup_code(
    code: str = typer.Argument(..., help="DTC code (e.g., P0171)"),
    lang: Language = typer.Option(Language.HU, "--lang", "-l", help="Output language"),
    format: OutputFormat = typer.Option(OutputFormat.TABLE, "--format", "-f", help="Output format"),
) -> None:
    """
    Look up a specific DTC code.

    Example: diagtool lookup P0171
    """
    code = code.upper().strip()

    if not validate_dtc_code(code):
        console.print(f"[red]Invalid DTC format: {code}[/red]")
        console.print("[dim]Valid format: [P/B/C/U] + 4 digits (e.g., P0171)[/dim]")
        raise typer.Exit(1)

    db = DTCDatabase()
    entry = db.get_code(code)

    if not entry:
        console.print(f"[yellow]DTC code not found: {code}[/yellow]")
        raise typer.Exit(1)

    if format == OutputFormat.JSON:
        console.print(json.dumps(entry, ensure_ascii=False, indent=2))
        return

    # Pretty table output
    severity = entry.get("severity", "medium")
    severity_color = SEVERITY_COLORS.get(severity, "white")

    desc = entry.get("description_hu" if lang == Language.HU else "description_en", "")
    desc_alt = entry.get("description_en" if lang == Language.HU else "description_hu", "")

    content_lines = []
    if desc:
        content_lines.append(f"[bold]{desc}[/bold]")
    if desc_alt and desc_alt != desc:
        content_lines.append(f"[dim]{desc_alt}[/dim]")

    content_lines.append("")
    content_lines.append(f"[bold]Category:[/bold] {get_language_text(CATEGORY_NAMES.get(entry.get('category', ''), {}), lang)}")
    content_lines.append(f"[bold]System:[/bold] {entry.get('system', 'N/A')}")
    content_lines.append(f"[bold]Severity:[/bold] [{severity_color}]{get_language_text(SEVERITY_NAMES.get(severity, {}), lang)}[/{severity_color}]")
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


@app.command("search")
def search_codes(
    query: str = typer.Argument(..., help="Search query"),
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Filter by category (powertrain, body, chassis, network)"),
    severity: Optional[str] = typer.Option(None, "--severity", "-s", help="Filter by severity (critical, high, medium, low)"),
    limit: int = typer.Option(20, "--limit", "-l", help="Maximum results"),
    lang: Language = typer.Option(Language.HU, "--lang", help="Output language"),
    format: OutputFormat = typer.Option(OutputFormat.TABLE, "--format", "-f", help="Output format"),
) -> None:
    """
    Search DTC codes by text.

    Example: diagtool search "oxygen sensor"
    """
    db = DTCDatabase()
    results = db.search_codes(query, category=category, severity=severity, limit=limit)

    if not results:
        console.print(f"[yellow]No results for: {query}[/yellow]")
        raise typer.Exit(0)

    if format == OutputFormat.JSON:
        output = {"query": query, "count": len(results), "codes": results}
        console.print(json.dumps(output, ensure_ascii=False, indent=2))
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

    for entry in results:
        severity = entry.get("severity", "medium")
        severity_color = SEVERITY_COLORS.get(severity, "white")
        desc = entry.get("description_hu" if lang == Language.HU else "description_en", "N/A")

        table.add_row(
            entry.get("code", "???"),
            desc[:45] if desc else "N/A",
            get_language_text(CATEGORY_NAMES.get(entry.get("category", ""), {}), lang)[:12],
            f"[{severity_color}]{get_language_text(SEVERITY_NAMES.get(severity, {}), lang)}[/{severity_color}]",
            (entry.get("system") or "N/A")[:18],
        )

    console.print(table)


@app.command("related")
def find_related(
    code: str = typer.Argument(..., help="DTC code to find related codes for"),
    lang: Language = typer.Option(Language.HU, "--lang", "-l", help="Output language"),
    format: OutputFormat = typer.Option(OutputFormat.TABLE, "--format", "-f", help="Output format"),
) -> None:
    """
    Find DTC codes related to a specific code.

    Example: diagtool related P0171
    """
    code = code.upper().strip()

    if not validate_dtc_code(code):
        console.print(f"[red]Invalid DTC format: {code}[/red]")
        raise typer.Exit(1)

    db = DTCDatabase()
    entry = db.get_code(code)

    if not entry:
        console.print(f"[yellow]DTC code not found: {code}[/yellow]")
        raise typer.Exit(1)

    related = db.get_related_codes(code)
    related_code_strs = entry.get("related_codes", [])

    if not related and not related_code_strs:
        console.print(f"[yellow]No related codes found for: {code}[/yellow]")
        raise typer.Exit(0)

    if format == OutputFormat.JSON:
        output = {
            "code": code,
            "related_codes": related_code_strs,
            "related_details": related,
        }
        console.print(json.dumps(output, ensure_ascii=False, indent=2))
        return

    console.print(f"\n[bold]Related codes for {code}:[/bold]\n")

    if related:
        table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
        table.add_column("Code", style="bold", width=8)
        table.add_column("Description", width=50)
        table.add_column("Severity", width=10)

        for rel_entry in related:
            severity = rel_entry.get("severity", "medium")
            severity_color = SEVERITY_COLORS.get(severity, "white")
            desc = rel_entry.get("description_hu" if lang == Language.HU else "description_en", "N/A")

            table.add_row(
                rel_entry.get("code", "???"),
                desc[:50] if desc else "N/A",
                f"[{severity_color}]{get_language_text(SEVERITY_NAMES.get(severity, {}), lang)}[/{severity_color}]",
            )

        console.print(table)
    else:
        console.print(f"[dim]Related codes (not in database): {', '.join(related_code_strs)}[/dim]")


# =============================================================================
# Diagnosis Commands
# =============================================================================


@app.command("diagnose")
def run_diagnosis(
    codes: Optional[str] = typer.Option(None, "--codes", "-c", help="DTC codes comma-separated (e.g., P0171,P0101)"),
    symptoms: Optional[str] = typer.Option(None, "--symptoms", "-s", help="Symptom description"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive mode with prompts"),
    make: Optional[str] = typer.Option(None, "--make", "-m", help="Vehicle make"),
    model: Optional[str] = typer.Option(None, "--model", help="Vehicle model"),
    year: Optional[int] = typer.Option(None, "--year", "-y", help="Vehicle year"),
    lang: Language = typer.Option(Language.HU, "--lang", "-l", help="Output language"),
    format: OutputFormat = typer.Option(OutputFormat.TABLE, "--format", "-f", help="Output format"),
) -> None:
    """
    Run vehicle diagnosis based on DTC codes and symptoms.

    Examples:
        diagtool diagnose --codes P0171,P0101 --symptoms "motor vibral"
        diagtool diagnose --interactive
    """
    db = DTCDatabase()

    if interactive:
        _run_interactive_diagnosis(db, lang)
        return

    if not codes and not symptoms:
        console.print("[red]Error: Provide --codes and/or --symptoms, or use --interactive[/red]")
        raise typer.Exit(1)

    # Parse codes
    code_list = []
    if codes:
        code_list = [c.strip().upper() for c in codes.split(",") if c.strip()]
        invalid = [c for c in code_list if not validate_dtc_code(c)]
        if invalid:
            console.print(f"[yellow]Warning: Invalid codes skipped: {', '.join(invalid)}[/yellow]")
            code_list = [c for c in code_list if validate_dtc_code(c)]

    # Lookup codes
    found_codes = []
    not_found = []
    for code in code_list:
        entry = db.get_code(code)
        if entry:
            found_codes.append(entry)
        else:
            not_found.append(code)

    # Search by symptoms
    symptom_results = []
    if symptoms:
        symptom_results = db.search_by_symptoms(symptoms, limit=10)

    if format == OutputFormat.JSON:
        output = {
            "diagnosis": {
                "input": {
                    "codes": code_list,
                    "symptoms": symptoms,
                    "vehicle": {"make": make, "model": model, "year": year} if any([make, model, year]) else None,
                },
                "found_codes": found_codes,
                "not_found": not_found,
                "symptom_matches": [
                    {"score": score, "code": entry} for score, entry in symptom_results
                ],
                "timestamp": datetime.now().isoformat(),
            }
        }
        console.print(json.dumps(output, ensure_ascii=False, indent=2))
        return

    # Pretty output
    console.print()
    header = "[bold blue]DIAGNOSIS RESULTS[/bold blue]"
    if make or model or year:
        header += f" - {make or ''} {model or ''} {year or ''}"
    console.print(Panel(header, box=box.DOUBLE))

    if symptoms:
        console.print(f"\n[bold]Symptoms:[/bold] {symptoms}")

    if not_found:
        console.print(f"\n[yellow]Codes not in database: {', '.join(not_found)}[/yellow]")

    if found_codes:
        console.print(f"\n[bold]DTC Code Analysis ({len(found_codes)} codes):[/bold]")

        # Group by severity
        for severity_level in ["critical", "high", "medium", "low"]:
            codes_at_level = [c for c in found_codes if c.get("severity") == severity_level]
            if codes_at_level:
                color = SEVERITY_COLORS.get(severity_level, "white")
                console.print(f"\n[bold {color}]{get_language_text(SEVERITY_NAMES.get(severity_level, {}), lang).upper()}:[/bold {color}]")

                for entry in codes_at_level:
                    desc = entry.get("description_hu" if lang == Language.HU else "description_en", "")
                    console.print(f"  [{color}]{entry.get('code')}[/{color}] - {desc}")

                    if entry.get("possible_causes"):
                        console.print(f"    [dim]Causes: {'; '.join(entry.get('possible_causes', [])[:3])}[/dim]")

    if symptom_results:
        console.print(f"\n[bold]Symptom-Based Matches:[/bold]")

        table = Table(box=box.SIMPLE, show_header=True)
        table.add_column("Score", justify="right", width=6)
        table.add_column("Code", width=8)
        table.add_column("Description", width=50)

        for score, entry in symptom_results[:10]:
            if entry.get("code") not in code_list:  # Avoid duplicates
                desc = entry.get("description_hu" if lang == Language.HU else "description_en", "")
                table.add_row(f"{score:.1f}", entry.get("code", ""), desc[:50] if desc else "")

        console.print(table)

    # Recommendations
    if found_codes:
        console.print("\n[bold]Recommendations:[/bold]")
        console.print("  1. Address critical and high severity codes first")
        console.print("  2. Follow diagnostic steps in order")
        console.print("  3. Check related codes for root cause analysis")

        if any(c.get("severity") == "critical" for c in found_codes):
            console.print(
                "\n[bold red]WARNING: Critical fault code detected! "
                "Immediate professional inspection recommended.[/bold red]"
            )


def _run_interactive_diagnosis(db: DTCDatabase, lang: Language) -> None:
    """Run interactive diagnosis wizard."""
    console.print()
    console.print(Panel(
        "[bold blue]AutoCognitix Interactive Diagnostics[/bold blue]\n"
        "Step-by-step guided diagnosis.",
        box=box.DOUBLE,
    ))

    try:
        # Step 1: Vehicle info
        console.print("\n[bold cyan]Step 1: Vehicle Information[/bold cyan]")

        make = Prompt.ask("Make (e.g., Volkswagen)", default="", console=console)
        model = Prompt.ask("Model (e.g., Golf)", default="", console=console)
        year_str = Prompt.ask("Year", default="", console=console)
        year = int(year_str) if year_str.isdigit() else None

        # Step 2: DTC codes
        console.print("\n[bold cyan]Step 2: DTC Codes[/bold cyan]")
        console.print("[dim]Enter codes one by one. Press Enter without input when done.[/dim]")

        code_list = []
        while True:
            code_input = Prompt.ask("DTC Code", default="", console=console)
            if not code_input:
                break

            code_upper = code_input.upper().strip()
            if validate_dtc_code(code_upper):
                if code_upper not in code_list:
                    code_list.append(code_upper)
                    entry = db.get_code(code_upper)
                    if entry:
                        desc = entry.get("description_hu" if lang == Language.HU else "description_en", "")
                        console.print(f"  [green]+ {code_upper}[/green] - {desc[:50]}")
                    else:
                        console.print(f"  [yellow]+ {code_upper}[/yellow] (not in database)")
            else:
                console.print(f"  [red]Invalid format: {code_input}[/red]")

        # Step 3: Symptoms
        console.print("\n[bold cyan]Step 3: Symptom Description[/bold cyan]")
        symptoms = Prompt.ask("Describe the symptoms", default="", console=console)

        # Step 4: Confirmation
        console.print("\n[bold cyan]Step 4: Summary[/bold cyan]")

        summary = []
        if make or model or year:
            summary.append(f"[bold]Vehicle:[/bold] {make} {model} {year or ''}")
        if code_list:
            summary.append(f"[bold]DTC Codes:[/bold] {', '.join(code_list)}")
        if symptoms:
            summary.append(f"[bold]Symptoms:[/bold] {symptoms[:80]}...")

        console.print(Panel("\n".join(summary) or "No data entered", title="Summary"))

        if not code_list and not symptoms:
            console.print("[yellow]No codes or symptoms provided. Exiting.[/yellow]")
            return

        if not Confirm.ask("Proceed with diagnosis?", default=True, console=console):
            console.print("[dim]Cancelled.[/dim]")
            return

        # Step 5: Run diagnosis
        console.print("\n[bold cyan]Step 5: Analysis[/bold cyan]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Analyzing...", total=None)

            found_codes = []
            for code in code_list:
                entry = db.get_code(code)
                if entry:
                    found_codes.append(entry)

            symptom_results = []
            if symptoms:
                symptom_results = db.search_by_symptoms(symptoms, limit=5)

        # Display results
        if found_codes:
            console.print(f"\n[bold]Found {len(found_codes)} codes in database:[/bold]")
            for entry in found_codes:
                severity = entry.get("severity", "medium")
                color = SEVERITY_COLORS.get(severity, "white")
                desc = entry.get("description_hu" if lang == Language.HU else "description_en", "")
                console.print(f"  [{color}]{entry.get('code')}[/{color}] - {desc}")

        if symptom_results:
            console.print(f"\n[bold]Symptom-based suggestions:[/bold]")
            for score, entry in symptom_results:
                if entry.get("code") not in code_list:
                    desc = entry.get("description_hu" if lang == Language.HU else "description_en", "")
                    console.print(f"  {entry.get('code')} (score: {score:.1f}) - {desc[:40]}")

        # Export option
        if Confirm.ask("\nExport results to JSON?", default=False, console=console):
            output = {
                "diagnosis": {
                    "vehicle": {"make": make, "model": model, "year": year},
                    "codes": code_list,
                    "symptoms": symptoms,
                    "found_codes": found_codes,
                    "symptom_matches": [{"score": s, "code": e} for s, e in symptom_results],
                    "timestamp": datetime.now().isoformat(),
                }
            }
            filename = f"diagnosis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            console.print(f"[green]Saved: {filename}[/green]")

        console.print("\n[dim]Diagnosis complete.[/dim]")

    except KeyboardInterrupt:
        console.print("\n[dim]Cancelled.[/dim]")


# =============================================================================
# Vehicle Commands
# =============================================================================


@vehicles_app.command("list")
def list_makes(
    format: OutputFormat = typer.Option(OutputFormat.TABLE, "--format", "-f", help="Output format"),
) -> None:
    """
    List all vehicle makes in the database.

    Example: diagtool vehicles list
    """
    db = VehicleDatabase()
    makes = db.get_makes()

    if not makes:
        console.print("[yellow]No vehicles in database.[/yellow]")
        console.print("[dim]Run import scripts to populate vehicle data.[/dim]")
        raise typer.Exit(0)

    if format == OutputFormat.JSON:
        console.print(json.dumps({"makes": makes, "count": len(makes)}, indent=2))
        return

    console.print(f"\n[bold]Vehicle Makes ({len(makes)}):[/bold]\n")

    # Display in columns
    columns = 4
    for i in range(0, len(makes), columns):
        row = makes[i:i+columns]
        console.print("  " + "  ".join(f"{m:<20}" for m in row))


@vehicles_app.command("models")
def list_models(
    make: str = typer.Argument(..., help="Vehicle make (e.g., VW, Volkswagen)"),
    format: OutputFormat = typer.Option(OutputFormat.TABLE, "--format", "-f", help="Output format"),
) -> None:
    """
    List models for a specific vehicle make.

    Example: diagtool vehicles models VW
    """
    db = VehicleDatabase()
    models = db.get_models(make)

    if not models:
        console.print(f"[yellow]No models found for: {make}[/yellow]")
        raise typer.Exit(0)

    if format == OutputFormat.JSON:
        console.print(json.dumps({"make": make, "models": models, "count": len(models)}, indent=2))
        return

    table = Table(
        title=f"Models for {make} ({len(models)} found)",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )

    table.add_column("ID", width=25)
    table.add_column("Make", width=15)
    table.add_column("Model", width=20)
    table.add_column("DTC Codes", justify="right", width=10)
    table.add_column("Signals", justify="right", width=10)

    for m in models:
        table.add_row(
            m.get("id", ""),
            m.get("make", ""),
            m.get("model", ""),
            str(m.get("dtc_count", 0)),
            str(m.get("signal_count", 0)),
        )

    console.print(table)


@vin_app.command("decode")
def decode_vin(
    vin: str = typer.Argument(..., help="VIN to decode (17 characters)"),
    format: OutputFormat = typer.Option(OutputFormat.TABLE, "--format", "-f", help="Output format"),
) -> None:
    """
    Decode a VIN using NHTSA API.

    Example: diagtool vin decode WVWZZZ3CZWE123456
    """
    vin = vin.strip().upper()

    if len(vin) != 17:
        console.print(f"[red]Invalid VIN length: {len(vin)} (expected 17)[/red]")
        raise typer.Exit(1)

    async def _decode():
        try:
            # Try to use NHTSA service
            from backend.app.services.nhtsa_service import NHTSAService

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task("Decoding VIN...", total=None)

                async with NHTSAService() as service:
                    result = await service.decode_vin(vin)
                    return result

        except ImportError:
            console.print("[yellow]NHTSA service not available. Using fallback.[/yellow]")
            return None
        except Exception as e:
            console.print(f"[red]VIN decode error: {e}[/red]")
            return None

    result = asyncio.run(_decode())

    if result is None:
        console.print("[red]Failed to decode VIN[/red]")
        raise typer.Exit(1)

    if format == OutputFormat.JSON:
        console.print(json.dumps(result.model_dump() if hasattr(result, 'model_dump') else dict(result), indent=2))
        return

    # Pretty output
    console.print(Panel(f"[bold]VIN: {vin}[/bold]", box=box.DOUBLE))

    data = result.model_dump() if hasattr(result, 'model_dump') else result

    table = Table(box=box.SIMPLE, show_header=False)
    table.add_column("Field", style="bold", width=25)
    table.add_column("Value", width=40)

    fields = [
        ("Make", "make"),
        ("Model", "model"),
        ("Year", "model_year"),
        ("Body Class", "body_class"),
        ("Vehicle Type", "vehicle_type"),
        ("Plant Country", "plant_country"),
        ("Manufacturer", "manufacturer"),
        ("Engine Cylinders", "engine_cylinders"),
        ("Engine Displacement (L)", "engine_displacement_l"),
        ("Fuel Type", "fuel_type_primary"),
        ("Transmission", "transmission_style"),
        ("Drive Type", "drive_type"),
    ]

    for label, key in fields:
        value = data.get(key, "")
        if value:
            table.add_row(label, str(value))

    console.print(table)


# =============================================================================
# Statistics Commands
# =============================================================================


@stats_app.callback(invoke_without_command=True)
def stats_default(
    ctx: typer.Context,
    format: OutputFormat = typer.Option(OutputFormat.TABLE, "--format", "-f", help="Output format"),
) -> None:
    """
    Show overall database statistics.

    Example: diagtool stats
    """
    if ctx.invoked_subcommand is not None:
        return

    dtc_db = DTCDatabase()
    vehicle_db = VehicleDatabase()
    trans_cache = TranslationCache()

    dtc_stats = dtc_db.get_statistics()
    trans_stats = trans_cache.get_statistics()

    if format == OutputFormat.JSON:
        output = {
            "dtc": dtc_stats,
            "vehicles": {"total": vehicle_db.total_vehicles},
            "translations": trans_stats,
        }
        console.print(json.dumps(output, indent=2))
        return

    console.print(Panel("[bold]AutoCognitix Database Statistics[/bold]", box=box.DOUBLE))

    # Overview table
    overview = Table(title="Overview", box=box.ROUNDED, show_header=False)
    overview.add_column("Metric", style="bold", width=30)
    overview.add_column("Value", justify="right", width=15)

    overview.add_row("Total DTC Codes", str(dtc_stats["total_codes"]))
    overview.add_row("Translated Codes", f"{dtc_stats['translated']} ({dtc_stats['translation_percentage']}%)")
    overview.add_row("Codes with Symptoms", str(dtc_stats["with_symptoms"]))
    overview.add_row("Total Vehicles", str(vehicle_db.total_vehicles))
    overview.add_row("Translation Cache", str(trans_stats["cached_translations"]))

    console.print(overview)

    # Category breakdown
    cat_table = Table(title="Categories", box=box.SIMPLE)
    cat_table.add_column("Category", width=20)
    cat_table.add_column("Count", justify="right", width=10)

    for cat, count in sorted(dtc_stats["categories"].items(), key=lambda x: -x[1]):
        cat_name = CATEGORY_NAMES.get(cat, {}).get("en", cat)
        cat_table.add_row(cat_name, str(count))

    console.print(cat_table)

    # Severity breakdown
    sev_table = Table(title="Severity Levels", box=box.SIMPLE)
    sev_table.add_column("Severity", width=15)
    sev_table.add_column("Count", justify="right", width=10)

    for sev, count in sorted(dtc_stats["severities"].items(), key=lambda x: -x[1]):
        color = SEVERITY_COLORS.get(sev, "white")
        sev_name = SEVERITY_NAMES.get(sev, {}).get("en", sev)
        sev_table.add_row(f"[{color}]{sev_name}[/{color}]", str(count))

    console.print(sev_table)


@stats_app.command("dtc")
def stats_dtc(
    format: OutputFormat = typer.Option(OutputFormat.TABLE, "--format", "-f", help="Output format"),
) -> None:
    """
    Show detailed DTC code statistics.

    Example: diagtool stats dtc
    """
    db = DTCDatabase()
    stats = db.get_statistics()

    if format == OutputFormat.JSON:
        console.print(json.dumps(stats, indent=2))
        return

    console.print(Panel("[bold]DTC Code Statistics[/bold]", box=box.DOUBLE))

    # Top systems
    sys_table = Table(title="Top 15 Systems", box=box.ROUNDED)
    sys_table.add_column("System", width=40)
    sys_table.add_column("Count", justify="right", width=10)

    for sys_name, count in list(stats["systems"].items())[:15]:
        sys_table.add_row(sys_name[:40], str(count))

    console.print(sys_table)

    # Sources
    if stats["sources"]:
        src_table = Table(title="Data Sources", box=box.SIMPLE)
        src_table.add_column("Source", width=20)
        src_table.add_column("Codes", justify="right", width=10)

        for src, count in sorted(stats["sources"].items(), key=lambda x: -x[1]):
            src_table.add_row(src, str(count))

        console.print(src_table)


@stats_app.command("translations")
def stats_translations(
    format: OutputFormat = typer.Option(OutputFormat.TABLE, "--format", "-f", help="Output format"),
) -> None:
    """
    Show translation statistics.

    Example: diagtool stats translations
    """
    trans_cache = TranslationCache()
    stats = trans_cache.get_statistics()

    if format == OutputFormat.JSON:
        console.print(json.dumps(stats, indent=2))
        return

    console.print(Panel("[bold]Translation Statistics[/bold]", box=box.DOUBLE))

    table = Table(box=box.ROUNDED, show_header=False)
    table.add_column("Metric", style="bold", width=30)
    table.add_column("Value", justify="right", width=15)

    table.add_row("Total DTC Codes", str(stats["total_codes"]))
    table.add_row("Translated in Database", str(stats["translated_in_database"]))
    table.add_row("Translation Cache Size", str(stats["cached_translations"]))
    table.add_row("Translation Coverage", f"{stats['translation_percentage']}%")
    table.add_row("Pending Translations", str(stats["pending"]))

    console.print(table)

    # Progress bar
    progress_pct = stats["translation_percentage"]
    bar_width = 40
    filled = int(bar_width * progress_pct / 100)
    bar = "[green]" + "=" * filled + "[/green]" + "[dim]" + "-" * (bar_width - filled) + "[/dim]"
    console.print(f"\nProgress: [{bar}] {progress_pct}%")


# =============================================================================
# Translation Commands
# =============================================================================


@app.command("translate")
def translate_code(
    code: str = typer.Argument(..., help="DTC code to translate (e.g., P0171)"),
    provider: str = typer.Option("deepseek", "--provider", "-p", help="LLM provider (deepseek, groq, ollama)"),
) -> None:
    """
    Translate a single DTC code description to Hungarian.

    Example: diagtool translate P0171
    """
    code = code.upper().strip()

    if not validate_dtc_code(code):
        console.print(f"[red]Invalid DTC format: {code}[/red]")
        raise typer.Exit(1)

    db = DTCDatabase()
    entry = db.get_code(code)

    if not entry:
        console.print(f"[yellow]DTC code not found: {code}[/yellow]")
        raise typer.Exit(1)

    if entry.get("description_hu"):
        console.print(f"[green]Already translated:[/green]")
        console.print(f"  EN: {entry.get('description_en', 'N/A')}")
        console.print(f"  HU: {entry.get('description_hu')}")
        return

    # Check translation cache
    cache = TranslationCache()
    cached = cache.get(code)
    if cached:
        console.print(f"[green]Found in cache:[/green]")
        console.print(f"  EN: {entry.get('description_en', 'N/A')}")
        console.print(f"  HU: {cached}")
        return

    console.print(f"[yellow]Translation needed for {code}[/yellow]")
    console.print(f"  EN: {entry.get('description_en', 'N/A')}")
    console.print(f"\n[dim]Use scripts/translate_to_hungarian.py to translate codes.[/dim]")
    console.print(f"[dim]  python scripts/translate_to_hungarian.py --translate --provider {provider} --limit 1[/dim]")


@app.command("translate-batch")
def translate_batch(
    limit: int = typer.Option(100, "--limit", "-l", help="Maximum codes to translate"),
    provider: str = typer.Option("deepseek", "--provider", "-p", help="LLM provider"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be translated without translating"),
) -> None:
    """
    Translate multiple DTC codes in batch.

    Example: diagtool translate-batch --limit 100 --provider groq
    """
    db = DTCDatabase()
    cache = TranslationCache()

    # Find codes needing translation
    pending = []
    for entry in db.all_codes:
        code = entry.get("code", "")
        if not entry.get("description_hu") and not cache.get(code):
            if entry.get("description_en"):
                pending.append(entry)

    if not pending:
        console.print("[green]All codes are already translated![/green]")
        return

    to_translate = pending[:limit]

    console.print(f"\n[bold]Batch Translation[/bold]")
    console.print(f"  Pending translations: {len(pending)}")
    console.print(f"  Will translate: {len(to_translate)}")
    console.print(f"  Provider: {provider}")

    if dry_run:
        console.print("\n[dim]Codes to translate (dry run):[/dim]")
        for entry in to_translate[:20]:
            console.print(f"  {entry.get('code')}: {entry.get('description_en', '')[:50]}")
        if len(to_translate) > 20:
            console.print(f"  ... and {len(to_translate) - 20} more")
        return

    console.print(f"\n[dim]Run the dedicated translation script:[/dim]")
    console.print(f"  python scripts/translate_to_hungarian.py --translate --provider {provider} --limit {limit}")


# =============================================================================
# Main Entry Point
# =============================================================================


def version_callback(value: bool) -> None:
    """Show version and exit."""
    if value:
        console.print("diagtool version 1.0.0")
        console.print("AutoCognitix Vehicle Diagnostics Platform")
        raise typer.Exit(0)


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    version: bool = typer.Option(
        False, "--version", "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version"
    ),
) -> None:
    """
    AutoCognitix Diagnostic Tool - Comprehensive CLI for vehicle diagnostics.

    Use --help with any command for detailed usage information.
    """
    # If no command is provided, show help
    if ctx.invoked_subcommand is None and not version:
        console.print(ctx.get_help())


if __name__ == "__main__":
    app()
