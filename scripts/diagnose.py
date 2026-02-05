#!/usr/bin/env python3
"""
AutoCognitix CLI Diagnostic Tool
================================

Command-line interface for vehicle diagnostics with DTC code lookup,
symptom-based search, and full diagnosis capabilities.

Usage:
    python diagnose.py lookup P0171
    python diagnose.py symptom "motor nehezen indul"
    python diagnose.py diagnose --codes P0171,P0300 --symptoms "..."
    python diagnose.py interactive

Author: AutoCognitix Team
"""

import csv
import io
import json
import re
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import typer
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.prompt import Confirm, Prompt
    from rich.table import Table
    from rich.text import Text
except ImportError:
    print("Hiba: Hianyzo fuggosegek. Telepitsd: pip install typer[all] rich")
    sys.exit(1)


# ============================================================================
# Constants and Configuration
# ============================================================================

# Project paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "dtc_codes"
DTC_DATA_FILE = DATA_DIR / "all_codes_merged.json"

# DTC code format validation
DTC_PATTERN = re.compile(r"^[PBCU]\d{4}$", re.IGNORECASE)

# Category mappings
CATEGORY_MAP = {
    "P": "powertrain",
    "B": "body",
    "C": "chassis",
    "U": "network",
}

CATEGORY_NAMES_HU = {
    "powertrain": "Hajtaslancz",
    "body": "Karosszeria",
    "chassis": "Alvaaz",
    "network": "Halozat/Kommunikacio",
}

SEVERITY_COLORS = {
    "critical": "red",
    "high": "orange1",
    "medium": "yellow",
    "low": "green",
}

SEVERITY_NAMES_HU = {
    "critical": "Kritikus",
    "high": "Magas",
    "medium": "Kozepes",
    "low": "Alacsony",
}

# Common vehicle makes for autocompletion
COMMON_MAKES = [
    "Audi", "BMW", "Citroen", "Dacia", "Fiat", "Ford", "Honda", "Hyundai",
    "Kia", "Mazda", "Mercedes-Benz", "Nissan", "Opel", "Peugeot", "Renault",
    "Seat", "Skoda", "Suzuki", "Toyota", "Volkswagen", "Volvo",
]


# ============================================================================
# Output Format Enum
# ============================================================================

class OutputFormat(str, Enum):
    """Output format options."""
    PRETTY = "pretty"
    JSON = "json"
    CSV = "csv"


# ============================================================================
# Data Loading
# ============================================================================

class DTCDatabase:
    """DTC code database handler with caching."""

    _instance: Optional["DTCDatabase"] = None
    _data: Optional[Dict[str, Any]] = None

    def __new__(cls) -> "DTCDatabase":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._data is None:
            self._load_data()

    def _load_data(self) -> None:
        """Load DTC data from JSON file."""
        if not DTC_DATA_FILE.exists():
            raise FileNotFoundError(
                f"DTC adatbazis nem talalhato: {DTC_DATA_FILE}\n"
                f"Futtasd elobb a seed scriptet."
            )

        with open(DTC_DATA_FILE, "r", encoding="utf-8") as f:
            self._data = json.load(f)

        # Build index by code
        self._code_index: Dict[str, Dict] = {}
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
            # Category filter
            if category and entry.get("category") != category:
                continue

            # Severity filter
            if severity and entry.get("severity") != severity:
                continue

            # Text search in code and descriptions
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
        make: Optional[str] = None,
        model: Optional[str] = None,
        year: Optional[int] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Search codes by symptom description."""
        # Tokenize symptoms
        symptom_words = set(
            word.lower()
            for word in re.findall(r"\w+", symptoms)
            if len(word) > 2
        )

        scored_results = []

        for code, entry in self._code_index.items():
            score = 0

            # Check symptoms array
            for symptom in entry.get("symptoms", []):
                symptom_lower = symptom.lower()
                for word in symptom_words:
                    if word in symptom_lower:
                        score += 2

            # Check possible causes
            for cause in entry.get("possible_causes", []):
                cause_lower = cause.lower()
                for word in symptom_words:
                    if word in cause_lower:
                        score += 1

            # Check descriptions
            desc_hu = (entry.get("description_hu") or "").lower()
            desc_en = (entry.get("description_en") or "").lower()
            for word in symptom_words:
                if word in desc_hu:
                    score += 1.5
                if word in desc_en:
                    score += 0.5

            # TODO: Filter by make/model/year when applicable_makes is populated

            if score > 0:
                scored_results.append((score, entry))

        # Sort by score descending
        scored_results.sort(key=lambda x: x[0], reverse=True)

        return [entry for _, entry in scored_results[:limit]]

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


# ============================================================================
# Output Formatters
# ============================================================================

class OutputFormatter:
    """Base class for output formatters."""

    def __init__(self, console: Console):
        self.console = console

    def format_code(self, code: Dict[str, Any]) -> str:
        """Format a single DTC code."""
        raise NotImplementedError

    def format_codes(self, codes: List[Dict[str, Any]]) -> str:
        """Format multiple DTC codes."""
        raise NotImplementedError

    def format_diagnosis(self, diagnosis: Dict[str, Any]) -> str:
        """Format diagnosis results."""
        raise NotImplementedError


class PrettyFormatter(OutputFormatter):
    """Rich terminal output formatter."""

    def format_code(self, code: Dict[str, Any]) -> None:
        """Display a single DTC code with rich formatting."""
        code_str = code.get("code", "???")
        severity = code.get("severity", "medium")
        severity_color = SEVERITY_COLORS.get(severity, "white")
        severity_hu = SEVERITY_NAMES_HU.get(severity, severity)

        # Header
        title = f"[bold]{code_str}[/bold] - {code.get('description_hu', code.get('description_en', 'N/A'))}"

        # Build content
        content_lines = []

        # English description if different
        desc_en = code.get("description_en", "")
        desc_hu = code.get("description_hu", "")
        if desc_en and desc_en != desc_hu:
            content_lines.append(f"[dim]EN: {desc_en}[/dim]")

        content_lines.append("")
        content_lines.append(
            f"[bold]Kategoria:[/bold] {CATEGORY_NAMES_HU.get(code.get('category', ''), code.get('category', 'N/A'))}"
        )
        content_lines.append(f"[bold]Rendszer:[/bold] {code.get('system', 'N/A')}")
        content_lines.append(
            f"[bold]Sulyossag:[/bold] [{severity_color}]{severity_hu}[/{severity_color}]"
        )

        # Symptoms
        symptoms = code.get("symptoms", [])
        if symptoms:
            content_lines.append("")
            content_lines.append("[bold]Tunetek:[/bold]")
            for symptom in symptoms[:5]:
                content_lines.append(f"  - {symptom}")

        # Possible causes
        causes = code.get("possible_causes", [])
        if causes:
            content_lines.append("")
            content_lines.append("[bold]Lehetseges okok:[/bold]")
            for cause in causes[:5]:
                content_lines.append(f"  - {cause}")

        # Diagnostic steps
        steps = code.get("diagnostic_steps", [])
        if steps:
            content_lines.append("")
            content_lines.append("[bold]Diagnosztikai lepesek:[/bold]")
            for i, step in enumerate(steps[:5], 1):
                content_lines.append(f"  {i}. {step}")

        # Related codes
        related = code.get("related_codes", [])
        if related:
            content_lines.append("")
            content_lines.append(f"[bold]Kapcsolodo kodok:[/bold] {', '.join(related[:10])}")

        # Sources
        sources = code.get("sources", [])
        if sources:
            content_lines.append("")
            content_lines.append(f"[dim]Forrasok: {', '.join(sources)}[/dim]")

        # Create panel
        panel = Panel(
            "\n".join(content_lines),
            title=title,
            border_style=severity_color,
            box=box.ROUNDED,
        )
        self.console.print(panel)

    def format_codes(self, codes: List[Dict[str, Any]], title: str = "Talalatok") -> None:
        """Display multiple DTC codes in a table."""
        if not codes:
            self.console.print("[yellow]Nincs talalat.[/yellow]")
            return

        table = Table(
            title=title,
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
        )

        table.add_column("Kod", style="bold", width=8)
        table.add_column("Leiras", width=40)
        table.add_column("Kategoria", width=12)
        table.add_column("Sulyossag", width=10)
        table.add_column("Rendszer", width=20)

        for code in codes:
            severity = code.get("severity", "medium")
            severity_color = SEVERITY_COLORS.get(severity, "white")

            table.add_row(
                code.get("code", "???"),
                (code.get("description_hu") or code.get("description_en", "N/A"))[:40],
                CATEGORY_NAMES_HU.get(code.get("category", ""), "")[:12],
                f"[{severity_color}]{SEVERITY_NAMES_HU.get(severity, severity)}[/{severity_color}]",
                (code.get("system") or "N/A")[:20],
            )

        self.console.print(table)
        self.console.print(f"\n[dim]Ossz. {len(codes)} talalat[/dim]")

    def format_diagnosis(
        self,
        codes: List[Dict[str, Any]],
        symptoms: str,
        vehicle: Optional[Dict[str, str]] = None,
    ) -> None:
        """Display full diagnosis results."""
        # Header
        self.console.print()
        header = "[bold blue]DIAGNOZIS EREDMENY[/bold blue]"
        if vehicle:
            vehicle_str = f"{vehicle.get('make', '')} {vehicle.get('model', '')} {vehicle.get('year', '')}"
            header += f" - {vehicle_str}"
        self.console.print(Panel(header, box=box.DOUBLE))

        # Symptoms
        self.console.print(f"\n[bold]Leirtak tunetek:[/bold]\n{symptoms}\n")

        # Codes analysis
        if codes:
            self.console.print(f"[bold]DTC kodok elemzese ({len(codes)} kod):[/bold]\n")

            # Group by severity
            critical = [c for c in codes if c.get("severity") == "critical"]
            high = [c for c in codes if c.get("severity") == "high"]
            medium = [c for c in codes if c.get("severity") == "medium"]
            low = [c for c in codes if c.get("severity") == "low"]

            if critical:
                self.console.print("[bold red]KRITIKUS HIBAK:[/bold red]")
                for code in critical:
                    self.format_code(code)

            if high:
                self.console.print("[bold orange1]MAGAS PRIORITASU:[/bold orange1]")
                for code in high:
                    self.format_code(code)

            if medium:
                self.console.print("[bold yellow]KOZEPES PRIORITASU:[/bold yellow]")
                for code in medium:
                    self.format_code(code)

            if low:
                self.console.print("[bold green]ALACSONY PRIORITASU:[/bold green]")
                for code in low:
                    self.format_code(code)

            # Recommendations
            self.console.print("\n[bold]Javasolt lepesek:[/bold]")
            self.console.print(
                "  1. Ellenorizze eloszor a kritikus es magas prioritasu hibakodokat"
            )
            self.console.print(
                "  2. Kovesse a diagnosztikai lepeseket sorrendben"
            )
            self.console.print(
                "  3. Ellenorizze a kapcsolodo kodokat is"
            )

            if any(c.get("severity") == "critical" for c in codes):
                self.console.print(
                    "\n[bold red]FIGYELEM: Kritikus hibakod talalhato! "
                    "Azonnali szakszervizi vizsgalat javasolt.[/bold red]"
                )
        else:
            self.console.print("[yellow]Nem talalhato ismert DTC kod.[/yellow]")


class JSONFormatter(OutputFormatter):
    """JSON output formatter."""

    def format_code(self, code: Dict[str, Any]) -> str:
        """Format a single DTC code as JSON."""
        return json.dumps(code, ensure_ascii=False, indent=2)

    def format_codes(self, codes: List[Dict[str, Any]], title: str = "") -> str:
        """Format multiple DTC codes as JSON."""
        output = {
            "count": len(codes),
            "codes": codes,
            "generated_at": datetime.now().isoformat(),
        }
        return json.dumps(output, ensure_ascii=False, indent=2)

    def format_diagnosis(
        self,
        codes: List[Dict[str, Any]],
        symptoms: str,
        vehicle: Optional[Dict[str, str]] = None,
    ) -> str:
        """Format diagnosis results as JSON."""
        output = {
            "diagnosis": {
                "vehicle": vehicle,
                "symptoms": symptoms,
                "codes": codes,
                "summary": {
                    "total_codes": len(codes),
                    "critical": len([c for c in codes if c.get("severity") == "critical"]),
                    "high": len([c for c in codes if c.get("severity") == "high"]),
                    "medium": len([c for c in codes if c.get("severity") == "medium"]),
                    "low": len([c for c in codes if c.get("severity") == "low"]),
                },
                "generated_at": datetime.now().isoformat(),
            }
        }
        return json.dumps(output, ensure_ascii=False, indent=2)


class CSVFormatter(OutputFormatter):
    """CSV output formatter."""

    def format_code(self, code: Dict[str, Any]) -> str:
        """Format a single DTC code as CSV."""
        return self.format_codes([code])

    def format_codes(self, codes: List[Dict[str, Any]], title: str = "") -> str:
        """Format multiple DTC codes as CSV."""
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            "code", "description_hu", "description_en", "category",
            "severity", "system", "symptoms", "possible_causes"
        ])

        # Data
        for code in codes:
            writer.writerow([
                code.get("code", ""),
                code.get("description_hu", ""),
                code.get("description_en", ""),
                code.get("category", ""),
                code.get("severity", ""),
                code.get("system", ""),
                "; ".join(code.get("symptoms", [])),
                "; ".join(code.get("possible_causes", [])),
            ])

        return output.getvalue()

    def format_diagnosis(
        self,
        codes: List[Dict[str, Any]],
        symptoms: str,
        vehicle: Optional[Dict[str, str]] = None,
    ) -> str:
        """Format diagnosis results as CSV."""
        # For diagnosis, include vehicle info in each row
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            "vehicle_make", "vehicle_model", "vehicle_year", "input_symptoms",
            "code", "description_hu", "category", "severity", "system"
        ])

        vehicle = vehicle or {}
        for code in codes:
            writer.writerow([
                vehicle.get("make", ""),
                vehicle.get("model", ""),
                vehicle.get("year", ""),
                symptoms,
                code.get("code", ""),
                code.get("description_hu", ""),
                code.get("category", ""),
                code.get("severity", ""),
                code.get("system", ""),
            ])

        return output.getvalue()


# ============================================================================
# CLI Application
# ============================================================================

app = typer.Typer(
    name="diagnose",
    help="AutoCognitix CLI diagnosztikai eszkoz",
    add_completion=True,
    rich_markup_mode="rich",
)
console = Console()


def validate_dtc_code(code: str) -> bool:
    """Validate DTC code format."""
    return bool(DTC_PATTERN.match(code.upper()))


def get_formatter(format_type: OutputFormat, console: Console) -> OutputFormatter:
    """Get the appropriate formatter for the output format."""
    if format_type == OutputFormat.JSON:
        return JSONFormatter(console)
    elif format_type == OutputFormat.CSV:
        return CSVFormatter(console)
    else:
        return PrettyFormatter(console)


def output_result(
    formatter: OutputFormatter,
    result: Any,
    format_type: OutputFormat,
) -> None:
    """Output formatted result."""
    if format_type == OutputFormat.PRETTY:
        # Pretty formatter prints directly
        pass
    else:
        # JSON and CSV formatters return strings
        console.print(result)


# ============================================================================
# CLI Commands
# ============================================================================

@app.command()
def lookup(
    code: str = typer.Argument(..., help="DTC kod (pl. P0171)"),
    format: OutputFormat = typer.Option(
        OutputFormat.PRETTY, "--format", "-f", help="Kimeneti formatum"
    ),
    related: bool = typer.Option(
        False, "--related", "-r", help="Kapcsolodo kodok megjelenitese"
    ),
) -> None:
    """
    DTC kod keresese az adatbazisban.

    Pelda: diagnose.py lookup P0171
    """
    code = code.upper().strip()

    # Validate format
    if not validate_dtc_code(code):
        console.print(f"[red]Ervenytelen DTC formatum: {code}[/red]")
        console.print("[dim]Ervenyes formatum: [P/B/C/U] + 4 szamjegy (pl. P0171)[/dim]")
        raise typer.Exit(1)

    try:
        db = DTCDatabase()
        formatter = get_formatter(format, console)

        entry = db.get_code(code)

        if not entry:
            console.print(f"[yellow]DTC kod nem talalhato: {code}[/yellow]")
            raise typer.Exit(1)

        if format == OutputFormat.PRETTY:
            formatter.format_code(entry)

            if related:
                related_codes = db.get_related_codes(code)
                if related_codes:
                    console.print("\n[bold]Kapcsolodo kodok:[/bold]")
                    formatter.format_codes(related_codes, title="Kapcsolodo kodok")
        else:
            if related:
                entry["related_code_details"] = db.get_related_codes(code)

            if format == OutputFormat.JSON:
                console.print(formatter.format_code(entry))
            else:
                console.print(formatter.format_code(entry))

    except FileNotFoundError as e:
        console.print(f"[red]Hiba: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def search(
    query: str = typer.Argument(..., help="Keresesei kifejezes"),
    category: Optional[str] = typer.Option(
        None, "--category", "-c",
        help="Szures kategoriara (powertrain, body, chassis, network)"
    ),
    severity: Optional[str] = typer.Option(
        None, "--severity", "-s",
        help="Szures sulyossagra (critical, high, medium, low)"
    ),
    limit: int = typer.Option(20, "--limit", "-l", help="Maximum talalatok szama"),
    format: OutputFormat = typer.Option(
        OutputFormat.PRETTY, "--format", "-f", help="Kimeneti formatum"
    ),
) -> None:
    """
    DTC kodok keresese szoveg alapjan.

    Pelda: diagnose.py search "levego" --category powertrain
    """
    try:
        db = DTCDatabase()
        formatter = get_formatter(format, console)

        results = db.search_codes(
            query=query,
            category=category,
            severity=severity,
            limit=limit,
        )

        if format == OutputFormat.PRETTY:
            formatter.format_codes(results, title=f"Kereses: '{query}'")
        else:
            output = formatter.format_codes(results, title=query)
            console.print(output)

    except FileNotFoundError as e:
        console.print(f"[red]Hiba: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def symptom(
    description: str = typer.Argument(..., help="Tunet leirasa magyarul"),
    make: Optional[str] = typer.Option(
        None, "--make", "-m", help="Jarmű gyarto (pl. VW, Audi)"
    ),
    model: Optional[str] = typer.Option(
        None, "--model", help="Jarmű modell (pl. Golf, A4)"
    ),
    year: Optional[int] = typer.Option(
        None, "--year", "-y", help="Evjarat (pl. 2018)"
    ),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum talalatok szama"),
    format: OutputFormat = typer.Option(
        OutputFormat.PRETTY, "--format", "-f", help="Kimeneti formatum"
    ),
) -> None:
    """
    DTC kodok keresese tunet alapjan.

    Pelda: diagnose.py symptom "motor nehezen indul hidegben"
    """
    try:
        db = DTCDatabase()
        formatter = get_formatter(format, console)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Kereses...", total=None)

            results = db.search_by_symptoms(
                symptoms=description,
                make=make,
                model=model,
                year=year,
                limit=limit,
            )

        if format == OutputFormat.PRETTY:
            console.print(f"\n[bold]Tunet:[/bold] {description}")
            if make or model or year:
                console.print(f"[bold]Jarmű:[/bold] {make or ''} {model or ''} {year or ''}")
            console.print()
            formatter.format_codes(results, title="Lehetseges DTC kodok")
        else:
            output = formatter.format_codes(results)
            console.print(output)

    except FileNotFoundError as e:
        console.print(f"[red]Hiba: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def diagnose(
    codes: str = typer.Option(
        ..., "--codes", "-c", help="DTC kodok vesszovel elvalasztva (pl. P0171,P0300)"
    ),
    symptoms: str = typer.Option(
        ..., "--symptoms", "-s", help="Tunet leirasa magyarul"
    ),
    make: Optional[str] = typer.Option(
        None, "--make", "-m", help="Jarmű gyarto (pl. VW, Audi)"
    ),
    model: Optional[str] = typer.Option(
        None, "--model", help="Jarmű modell (pl. Golf, A4)"
    ),
    year: Optional[int] = typer.Option(
        None, "--year", "-y", help="Evjarat (pl. 2018)"
    ),
    format: OutputFormat = typer.Option(
        OutputFormat.PRETTY, "--format", "-f", help="Kimeneti formatum"
    ),
) -> None:
    """
    Teljes diagnozis DTC kodok es tunetek alapjan.

    Pelda: diagnose.py diagnose --codes P0171,P0300 --symptoms "Motor nehezen indul"
    """
    try:
        db = DTCDatabase()
        formatter = get_formatter(format, console)

        # Parse codes
        code_list = [c.strip().upper() for c in codes.split(",") if c.strip()]

        # Validate codes
        invalid_codes = [c for c in code_list if not validate_dtc_code(c)]
        if invalid_codes:
            console.print(f"[yellow]Figyelmeztetes: Ervenytelen kodok: {', '.join(invalid_codes)}[/yellow]")
            code_list = [c for c in code_list if validate_dtc_code(c)]

        if not code_list:
            console.print("[red]Hiba: Nincs ervenyes DTC kod[/red]")
            raise typer.Exit(1)

        # Lookup codes
        found_codes = []
        not_found = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Kodok elemzese...", total=None)

            for code in code_list:
                entry = db.get_code(code)
                if entry:
                    found_codes.append(entry)
                else:
                    not_found.append(code)

        if not_found:
            console.print(f"[yellow]Nem talalhato: {', '.join(not_found)}[/yellow]")

        # Build vehicle info
        vehicle = None
        if make or model or year:
            vehicle = {"make": make or "", "model": model or "", "year": str(year) if year else ""}

        # Format output
        if format == OutputFormat.PRETTY:
            formatter.format_diagnosis(found_codes, symptoms, vehicle)
        else:
            output = formatter.format_diagnosis(found_codes, symptoms, vehicle)
            console.print(output)

    except FileNotFoundError as e:
        console.print(f"[red]Hiba: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def interactive() -> None:
    """
    Interaktiv diagnosztikai varazslo.

    Lepesrol lepesre vegigvezet a diagnosztikai folyamaton.
    """
    console.print()
    console.print(Panel(
        "[bold blue]AutoCognitix Interaktiv Diagnosztika[/bold blue]\n"
        "Lepesrol lepesre vegigvezetem a diagnosztikan.",
        box=box.DOUBLE,
    ))

    try:
        db = DTCDatabase()
        formatter = PrettyFormatter(console)

        # Step 1: Vehicle information
        console.print("\n[bold cyan]1. lepes: Jarmű adatok[/bold cyan]")

        make = Prompt.ask(
            "Gyarto",
            default="",
            console=console,
        )

        model = Prompt.ask(
            "Modell",
            default="",
            console=console,
        )

        year_str = Prompt.ask(
            "Evjarat",
            default="",
            console=console,
        )
        year = int(year_str) if year_str.isdigit() else None

        # Step 2: DTC codes
        console.print("\n[bold cyan]2. lepes: DTC kodok[/bold cyan]")
        console.print("[dim]Irja be a hibakodokat (pl. P0171, P0300). Ures sor = kesz.[/dim]")

        code_list = []
        while True:
            code_input = Prompt.ask("DTC kod", default="", console=console)
            if not code_input:
                break

            code_upper = code_input.upper().strip()
            if validate_dtc_code(code_upper):
                if code_upper not in code_list:
                    code_list.append(code_upper)
                    console.print(f"  [green]+ {code_upper}[/green]")
                else:
                    console.print(f"  [yellow]Mar hozzaadva: {code_upper}[/yellow]")
            else:
                console.print(f"  [red]Ervenytelen formatum: {code_input}[/red]")

        if code_list:
            console.print(f"\n[bold]Hozzaadott kodok:[/bold] {', '.join(code_list)}")

        # Step 3: Symptoms
        console.print("\n[bold cyan]3. lepes: Tunetek leirasa[/bold cyan]")
        console.print("[dim]Irja le a tapasztalt problemaket magyarul.[/dim]")

        symptoms = Prompt.ask(
            "Tunetek",
            default="",
            console=console,
        )

        # Step 4: Confirmation
        console.print("\n[bold cyan]4. lepes: Osszegzes[/bold cyan]")

        summary_lines = []
        if make or model or year:
            summary_lines.append(f"[bold]Jarmű:[/bold] {make} {model} {year or ''}")
        if code_list:
            summary_lines.append(f"[bold]DTC kodok:[/bold] {', '.join(code_list)}")
        if symptoms:
            summary_lines.append(f"[bold]Tunetek:[/bold] {symptoms[:100]}...")

        console.print(Panel("\n".join(summary_lines) or "Nincs adat megadva", title="Osszegzes"))

        if not code_list and not symptoms:
            console.print("[yellow]Nem adott meg sem kodot, sem tunetet. Kileps.[/yellow]")
            raise typer.Exit(0)

        if not Confirm.ask("Folytatja a diagnosztikát?", default=True, console=console):
            console.print("[dim]Megszakitva.[/dim]")
            raise typer.Exit(0)

        # Step 5: Run diagnosis
        console.print("\n[bold cyan]5. lepes: Diagnozis[/bold cyan]")

        found_codes = []
        symptom_matches = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Elemzes...", total=None)

            # Lookup entered codes
            for code in code_list:
                entry = db.get_code(code)
                if entry:
                    found_codes.append(entry)

            # Search by symptoms if provided
            if symptoms:
                symptom_matches = db.search_by_symptoms(
                    symptoms=symptoms,
                    make=make if make else None,
                    model=model if model else None,
                    year=year,
                    limit=5,
                )

        # Display results
        vehicle = None
        if make or model or year:
            vehicle = {"make": make, "model": model, "year": str(year) if year else ""}

        # Combine found codes with symptom matches (avoid duplicates)
        all_codes = found_codes.copy()
        found_code_ids = {c.get("code") for c in found_codes}
        for match in symptom_matches:
            if match.get("code") not in found_code_ids:
                all_codes.append(match)

        if all_codes:
            formatter.format_diagnosis(all_codes, symptoms or "N/A", vehicle)
        else:
            console.print("[yellow]Nem talalhato ismert DTC kod vagy releváns talalat.[/yellow]")

        # Step 6: Additional options
        console.print("\n[bold cyan]6. lepes: Tovabbi muveletek[/bold cyan]")

        if Confirm.ask("Exportalja az eredmenyt JSON-ba?", default=False, console=console):
            json_formatter = JSONFormatter(console)
            output = json_formatter.format_diagnosis(all_codes, symptoms or "", vehicle)

            output_file = Path(f"diagnosis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(output)
            console.print(f"[green]Mentve: {output_file}[/green]")

        console.print("\n[dim]Diagnosztika befejezve.[/dim]")

    except FileNotFoundError as e:
        console.print(f"[red]Hiba: {e}[/red]")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n[dim]Megszakitva.[/dim]")
        raise typer.Exit(0)


@app.command()
def stats() -> None:
    """
    Adatbazis statisztikak megjelenitese.
    """
    try:
        db = DTCDatabase()

        console.print(Panel("[bold]AutoCognitix DTC Adatbazis[/bold]", box=box.DOUBLE))

        # Metadata
        meta = db.metadata
        console.print(f"\n[bold]Metaadatok:[/bold]")
        console.print(f"  Generalva: {meta.get('generated_at', 'N/A')}")
        console.print(f"  Osszes kod: {db.total_codes}")
        console.print(f"  Forditott: {meta.get('translated', 'N/A')}")

        # Category breakdown
        categories = {}
        severities = {}
        systems = {}

        for entry in db._code_index.values():
            cat = entry.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

            sev = entry.get("severity", "unknown")
            severities[sev] = severities.get(sev, 0) + 1

            sys = entry.get("system", "N/A")
            systems[sys] = systems.get(sys, 0) + 1

        # Category table
        console.print("\n[bold]Kategoriak:[/bold]")
        cat_table = Table(box=box.SIMPLE)
        cat_table.add_column("Kategoria")
        cat_table.add_column("Darab", justify="right")

        for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
            cat_table.add_row(CATEGORY_NAMES_HU.get(cat, cat), str(count))
        console.print(cat_table)

        # Severity table
        console.print("\n[bold]Sulyossag:[/bold]")
        sev_table = Table(box=box.SIMPLE)
        sev_table.add_column("Sulyossag")
        sev_table.add_column("Darab", justify="right")

        for sev, count in sorted(severities.items(), key=lambda x: -x[1]):
            color = SEVERITY_COLORS.get(sev, "white")
            sev_table.add_row(
                f"[{color}]{SEVERITY_NAMES_HU.get(sev, sev)}[/{color}]",
                str(count)
            )
        console.print(sev_table)

        # Top systems
        console.print("\n[bold]Top 10 rendszer:[/bold]")
        sys_table = Table(box=box.SIMPLE)
        sys_table.add_column("Rendszer")
        sys_table.add_column("Darab", justify="right")

        for sys, count in sorted(systems.items(), key=lambda x: -x[1])[:10]:
            sys_table.add_row(sys[:40], str(count))
        console.print(sys_table)

    except FileNotFoundError as e:
        console.print(f"[red]Hiba: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def export(
    output_file: str = typer.Argument(..., help="Kimeneti fajl neve"),
    format: OutputFormat = typer.Option(
        OutputFormat.JSON, "--format", "-f", help="Kimeneti formatum (json/csv)"
    ),
    category: Optional[str] = typer.Option(
        None, "--category", "-c", help="Szures kategoriara"
    ),
    severity: Optional[str] = typer.Option(
        None, "--severity", "-s", help="Szures sulyossagra"
    ),
) -> None:
    """
    Teljes adatbazis vagy szurt resz exportalasa.

    Pelda: diagnose.py export output.json --format json --category powertrain
    """
    try:
        db = DTCDatabase()

        # Collect filtered codes
        codes = []
        for entry in db._code_index.values():
            if category and entry.get("category") != category:
                continue
            if severity and entry.get("severity") != severity:
                continue
            codes.append(entry)

        # Sort by code
        codes.sort(key=lambda x: x.get("code", ""))

        # Format output
        formatter = get_formatter(format, console)
        output_path = Path(output_file)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Exportalas...", total=None)

            if format == OutputFormat.JSON:
                output = {
                    "metadata": {
                        "exported_at": datetime.now().isoformat(),
                        "total_codes": len(codes),
                        "filters": {
                            "category": category,
                            "severity": severity,
                        },
                    },
                    "codes": codes,
                }
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(output, f, ensure_ascii=False, indent=2)
            else:
                output = formatter.format_codes(codes)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(output)

        console.print(f"[green]Exportalva: {output_path} ({len(codes)} kod)[/green]")

    except FileNotFoundError as e:
        console.print(f"[red]Hiba: {e}[/red]")
        raise typer.Exit(1)


@app.callback()
def main_callback() -> None:
    """
    AutoCognitix CLI diagnosztikai eszkoz.

    Hasznalja a --help kapcsolot a parancsok listazasahoz.
    """
    pass


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    app()
