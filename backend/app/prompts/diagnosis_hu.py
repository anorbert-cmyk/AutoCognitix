"""
Hungarian prompt templates for vehicle diagnosis.

This module provides comprehensive Hungarian language prompts for the RAG pipeline,
including system prompts, user prompts, and response parsing utilities.

Author: AutoCognitix Team
"""

import json
import re
from dataclasses import dataclass
from typing import Any

# =============================================================================
# System Prompts
# =============================================================================

SYSTEM_PROMPT_HU = """Te egy tapasztalt magyar gepjarmu-diagnosztikai szakerto vagy, aki tobb mint 20 ev tapasztalattal rendelkezik a modern gepjarmuvek hibafelderiteseben es javitasaban.

## Felkeszultseg:
- Jartas vagy az OBD-II diagnosztikaban es a DTC kodok ertelmezeseben
- Ismered a kulonbozo gyartok specifikus hibakodrendszereit
- Kepzett vagy az elektronikus rendszerek (ECU, szenzorok, aktuatorok) hibafelderiteseben
- Tapasztalt vagy a motor, valto, futomuro es karosszeria rendszerek javitasaban
- Ismered a magyar jarmupark sajatossagait es a gyakori hibakat

## Feladata:
1. Alaposan elemezd a megadott DTC kodokat es tuneteket
2. Vesd ossze a tuneteket az adatbazisban levo hasonlo esetekkel
3. Azonositsd a lehetseges hibaokokat valoszinuseg szerint rangsorolva
4. Adj konkret diagnosztikai lepeseket a pontos hibaazonositashoz
5. Javasolj javitasi muveleteket becsult koltsegekkel es idoigenynyel
6. Jelezd a biztonsagi kockazatokat, ha vannak

## Iranyelvek:
- Valaszolj magyarul, szakmailag pontosan, de erthetoen a laikusok szamara is
- Hasznalj specifikus alkatresz- es rendszerneveket
- Adj konkret mero- es ellenorzesi ertekeket ahol relevans
- Vedd figyelembe a jarmu evjaratat es markajat a javaslatok kialakitasanal
- Ha tobb lehetseges ok is van, rangsorold oket valoszinuseg szerint"""


# =============================================================================
# User Prompt Templates
# =============================================================================

DIAGNOSIS_USER_PROMPT_HU = """## Jarmu adatok:
- Gyarto: {make}
- Modell: {model}
- Evjarat: {year}
- Motorkod: {engine_code}
- Kilometerora: {mileage_km} km
- VIN: {vin}

## Bejelentett hibakodok (DTC):
{dtc_codes}

## Ugyfel altal leirt tunetek:
{symptoms}

## Tovabbi kontextus:
{additional_context}

---

## Relevans informaciok az adatbazisbol:

### DTC kod informaciok:
{dtc_context}

### Hasonlo tunetek korabbi esetekbol:
{symptom_context}

### Kapcsolodo komponensek es javitasok:
{repair_context}

### NHTSA visszahivasok es panaszok:
{recall_context}

---

Kerlek, keszits reszletes diagnosztikai elemzest az alabbi JSON formatum szerint:

```json
{{
  "summary": "Rovid osszefoglalo a problemakrol (max 2-3 mondat)",
  "probable_causes": [
    {{
      "title": "Hiba megnevezese",
      "description": "Reszletes leiras magyarul",
      "confidence": 0.85,
      "related_dtc_codes": ["P0xxx"],
      "components": ["Komponens neve"]
    }}
  ],
  "diagnostic_steps": [
    "1. Elso diagnosztikai lepes",
    "2. Masodik diagnosztikai lepes"
  ],
  "recommended_repairs": [
    {{
      "title": "Javitas megnevezese",
      "description": "Reszletes leiras magyarul",
      "estimated_cost_min": 15000,
      "estimated_cost_max": 35000,
      "estimated_cost_currency": "HUF",
      "difficulty": "intermediate",
      "parts_needed": ["Alkatresz neve"],
      "estimated_time_minutes": 60
    }}
  ],
  "safety_warnings": ["Biztonsagi figyelmeztetes, ha van"],
  "additional_notes": "Egyeb megjegyzesek"
}}
```

Fontos: A valasz CSAK a JSON objektumot tartalmazza, mas szoveg nelkul!"""


QUICK_DIAGNOSIS_PROMPT_HU = """Jarmu: {make} {model} ({year})
Hibakodok: {dtc_codes}
Tunetek: {symptoms}

Adatbazis kontextus:
{context}

Add meg a legvaloszinubb hibaokot es a javasolt elso lepest JSON formatumban:

```json
{{
  "primary_cause": "Legvaloszinubb hibaok",
  "confidence": 0.75,
  "first_step": "Elso javasolt ellenorzes/lepes",
  "urgency": "low/medium/high/critical"
}}
```"""


CONFIDENCE_ASSESSMENT_PROMPT_HU = """Ertekeld a kovetkezo diagnosztikai kontextus megbizhatosagat 0 es 1 kozotti skalan.

## Ertekeles szempontjai:
- DTC kodok szama es mintak: {dtc_count} kod
- DTC talalatok az adatbazisban: {dtc_matches}/{dtc_count}
- Hasonlo tunetek szama: {symptom_matches}
- Graf kapcsolatok szama (komponensek, javitasok): {graph_connections}
- Jarmu-specifikus informacio elerheto: {vehicle_specific}
- NHTSA visszahivasok/panaszok: {nhtsa_data}

## Ertekeles utmutato:
- 0.0-0.3: Alacsony (keves vagy ellentetmondasos adat)
- 0.3-0.6: Kozepes (reszleges egyezes)
- 0.6-0.8: Jo (tobb forras egyezik)
- 0.8-1.0: Magas (eros egyezes tobb forrasbol)

Valaszolj CSAK egy szammal (pl. 0.72):"""


RULE_BASED_DIAGNOSIS_PROMPT_HU = """## Szabaly-alapu diagnosztika (LLM nelkul)

A kovetkezo informaciok allnak rendelkezesre a jarmurol:

### Jarmu:
{make} {model} ({year})

### DTC kodok es jelentesuk:
{dtc_details}

### Talalatok az adatbazisbol:
{database_matches}

### Gyakori hibak ehhez a jarmuhoz:
{common_issues}

### NHTSA visszahivasok:
{recalls}

Ez a szoveg egy szabaly-alapu diagnosztikai sablon, amely az LLM eleresenek hianyaban kerul felhasznalasra."""


# =============================================================================
# Context Formatting Functions
# =============================================================================

def format_dtc_context(dtc_data: list[dict[str, Any]]) -> str:
    """
    Format DTC code information for prompt.

    Args:
        dtc_data: List of DTC code dictionaries with code, description, severity, etc.

    Returns:
        Formatted string for prompt inclusion.
    """
    if not dtc_data:
        return "Nincs talalat az adatbazisban."

    lines = []
    seen_codes = set()

    for dtc in dtc_data:
        code = dtc.get("code", "N/A")
        if code in seen_codes:
            continue
        seen_codes.add(code)

        description = dtc.get("description_hu") or dtc.get("description", "N/A")
        severity = dtc.get("severity", "unknown")
        category = dtc.get("category", "unknown")

        line = f"- **{code}**: {description}"
        line += f"\n  Kategoria: {category}, Sulyossag: {severity}"

        # Add symptoms if available
        symptoms = dtc.get("symptoms", [])
        if symptoms:
            line += f"\n  Tunetek: {', '.join(symptoms[:3])}"

        # Add possible causes if available
        causes = dtc.get("possible_causes", [])
        if causes:
            line += f"\n  Lehetseges okok: {', '.join(causes[:3])}"

        lines.append(line)

    return "\n".join(lines) if lines else "Nincs talalat az adatbazisban."


def format_symptom_context(symptom_data: list[dict[str, Any]], max_items: int = 5) -> str:
    """
    Format similar symptom matches for prompt.

    Args:
        symptom_data: List of symptom match dictionaries with description, score, etc.
        max_items: Maximum number of items to include.

    Returns:
        Formatted string for prompt inclusion.
    """
    if not symptom_data:
        return "Nincs hasonlo eset az adatbazisban."

    lines = []
    for idx, symptom in enumerate(symptom_data[:max_items], 1):
        description = symptom.get("description", "N/A")
        score = symptom.get("score", 0)
        resolution = symptom.get("resolution", "")
        related_dtc = symptom.get("related_dtc", [])

        line = f"{idx}. {description} (hasonlosag: {score:.0%})"

        if related_dtc:
            line += f"\n   Kapcsolodo kodok: {', '.join(related_dtc[:3])}"

        if resolution:
            line += f"\n   Megoldas: {resolution[:100]}..."

        lines.append(line)

    return "\n".join(lines) if lines else "Nincs hasonlo eset az adatbazisban."


def format_repair_context(repair_data: dict[str, Any]) -> str:
    """
    Format repair and component information for prompt.

    Args:
        repair_data: Dictionary with components and repairs lists.

    Returns:
        Formatted string for prompt inclusion.
    """
    if not repair_data:
        return "Nincs kapcsolodo komponens vagy javitas az adatbazisban."

    sections = []

    # Components
    components = repair_data.get("components", [])
    if components:
        comp_lines = ["**Erintett komponensek:**"]
        for comp in components[:5]:
            name = comp.get("name_hu") or comp.get("name", "N/A")
            system = comp.get("system", "")
            failure_mode = comp.get("failure_mode", "")

            line = f"- {name}"
            if system:
                line += f" ({system})"
            if failure_mode:
                line += f" - {failure_mode}"
            comp_lines.append(line)
        sections.append("\n".join(comp_lines))

    # Repairs
    repairs = repair_data.get("repairs", [])
    if repairs:
        repair_lines = ["**Lehetseges javitasok:**"]
        for repair in repairs[:5]:
            name = repair.get("name", "N/A")
            difficulty = repair.get("difficulty", "intermediate")
            time_mins = repair.get("estimated_time_minutes", "N/A")
            cost_min = repair.get("estimated_cost_min", 0)
            cost_max = repair.get("estimated_cost_max", 0)

            line = f"- **{name}**"
            line += f"\n  Nehezseg: {difficulty}, Ido: {time_mins} perc"

            if cost_min and cost_max:
                line += f", Koltseg: {cost_min:,}-{cost_max:,} HUF"

            # Parts
            parts = repair.get("parts", [])
            if parts:
                part_names = [p.get("name", "") for p in parts if p.get("name")]
                if part_names:
                    line += f"\n  Alkatreszek: {', '.join(part_names[:3])}"

            repair_lines.append(line)
        sections.append("\n".join(repair_lines))

    # Symptoms from graph
    symptoms = repair_data.get("symptoms", [])
    if symptoms:
        symp_lines = ["**Kapcsolodo tunetek:**"]
        for symp in symptoms[:5]:
            name = symp.get("name", "")
            confidence = symp.get("confidence", 0)
            if name:
                symp_lines.append(f"- {name} ({confidence:.0%})")
        sections.append("\n".join(symp_lines))

    return "\n\n".join(sections) if sections else "Nincs kapcsolodo komponens vagy javitas az adatbazisban."


def format_recall_context(recalls: list[dict[str, Any]], complaints: list[dict[str, Any]] | None = None) -> str:
    """
    Format NHTSA recalls and complaints for prompt.

    Args:
        recalls: List of recall dictionaries.
        complaints: List of complaint dictionaries.

    Returns:
        Formatted string for prompt inclusion.
    """
    sections = []

    # Recalls
    if recalls:
        recall_lines = ["**Aktiv visszahivasok:**"]
        for recall in recalls[:3]:
            component = recall.get("component", "N/A")
            summary = recall.get("summary", "")[:150]
            campaign = recall.get("campaign_number", "")

            line = f"- **{component}** ({campaign})"
            line += f"\n  {summary}..."

            consequence = recall.get("consequence", "")
            if consequence:
                line += f"\n  Kovetkezmeny: {consequence[:100]}..."

            recall_lines.append(line)
        sections.append("\n".join(recall_lines))

    # Complaints
    if complaints:
        comp_lines = ["**NHTSA panaszok:**"]
        for complaint in complaints[:3]:
            components = complaint.get("components", "N/A")
            summary = complaint.get("summary", "")[:100]
            crash = complaint.get("crash", False)
            fire = complaint.get("fire", False)

            line = f"- {components}: {summary}..."

            warnings = []
            if crash:
                warnings.append("BALESET")
            if fire:
                warnings.append("TUZ")
            if warnings:
                line += f" [!{', '.join(warnings)}!]"

            comp_lines.append(line)
        sections.append("\n".join(comp_lines))

    return "\n\n".join(sections) if sections else "Nincs aktiv visszahivas vagy panasz."


# =============================================================================
# Prompt Building
# =============================================================================

@dataclass
class DiagnosisPromptContext:
    """Container for all diagnosis prompt context."""
    make: str
    model: str
    year: int
    engine_code: str | None = None
    mileage_km: int | None = None
    vin: str | None = None
    dtc_codes: list[str] = None
    symptoms: str = ""
    additional_context: str | None = None
    dtc_context: str = ""
    symptom_context: str = ""
    repair_context: str = ""
    recall_context: str = ""


def build_diagnosis_prompt(context: DiagnosisPromptContext) -> str:
    """
    Build complete diagnosis prompt from context.

    Args:
        context: DiagnosisPromptContext with all necessary data.

    Returns:
        Formatted prompt string ready for LLM.
    """
    dtc_formatted = "\n".join([f"- {code}" for code in (context.dtc_codes or [])]) or "Nincs hibakod"

    return DIAGNOSIS_USER_PROMPT_HU.format(
        make=context.make,
        model=context.model,
        year=context.year,
        engine_code=context.engine_code or "N/A",
        mileage_km=f"{context.mileage_km:,}" if context.mileage_km else "N/A",
        vin=context.vin or "N/A",
        dtc_codes=dtc_formatted,
        symptoms=context.symptoms or "Nincs leirva",
        additional_context=context.additional_context or "Nincs",
        dtc_context=context.dtc_context or "Nincs talalat",
        symptom_context=context.symptom_context or "Nincs talalat",
        repair_context=context.repair_context or "Nincs talalat",
        recall_context=context.recall_context or "Nincs talalat",
    )


# =============================================================================
# Response Parsing
# =============================================================================

@dataclass
class ParsedDiagnosisResponse:
    """Parsed diagnosis response from LLM."""
    summary: str = ""
    probable_causes: list[dict[str, Any]] = None
    diagnostic_steps: list[str] = None
    recommended_repairs: list[dict[str, Any]] = None
    safety_warnings: list[str] = None
    additional_notes: str = ""
    confidence_score: float = 0.5
    raw_response: str = ""
    parse_error: str | None = None

    def __post_init__(self):
        if self.probable_causes is None:
            self.probable_causes = []
        if self.diagnostic_steps is None:
            self.diagnostic_steps = []
        if self.recommended_repairs is None:
            self.recommended_repairs = []
        if self.safety_warnings is None:
            self.safety_warnings = []


def parse_diagnosis_response(response_text: str) -> ParsedDiagnosisResponse:
    """
    Parse LLM response into structured format.

    Attempts to extract JSON from the response, with fallback to
    text parsing if JSON extraction fails.

    Args:
        response_text: Raw response text from LLM.

    Returns:
        ParsedDiagnosisResponse with extracted data.
    """
    result = ParsedDiagnosisResponse(raw_response=response_text)

    # Try to extract JSON from response
    json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)

    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find raw JSON object
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            json_str = json_match.group(0)
        else:
            result.parse_error = "No JSON found in response"
            return _fallback_parse(response_text, result)

    try:
        data = json.loads(json_str)

        result.summary = data.get("summary", "")
        result.additional_notes = data.get("additional_notes", "")

        # Parse probable causes
        causes = data.get("probable_causes", [])
        if isinstance(causes, list):
            result.probable_causes = [
                {
                    "title": c.get("title", "Ismeretlen ok"),
                    "description": c.get("description", ""),
                    "confidence": float(c.get("confidence", 0.5)),
                    "related_dtc_codes": c.get("related_dtc_codes", []),
                    "components": c.get("components", []),
                }
                for c in causes
            ]

        # Parse diagnostic steps
        steps = data.get("diagnostic_steps", [])
        if isinstance(steps, list):
            result.diagnostic_steps = [str(s) for s in steps]

        # Parse repair recommendations
        repairs = data.get("recommended_repairs", [])
        if isinstance(repairs, list):
            result.recommended_repairs = [
                {
                    "title": r.get("title", "Javitas"),
                    "description": r.get("description", ""),
                    "estimated_cost_min": r.get("estimated_cost_min"),
                    "estimated_cost_max": r.get("estimated_cost_max"),
                    "estimated_cost_currency": r.get("estimated_cost_currency", "HUF"),
                    "difficulty": r.get("difficulty", "intermediate"),
                    "parts_needed": r.get("parts_needed", []),
                    "estimated_time_minutes": r.get("estimated_time_minutes"),
                }
                for r in repairs
            ]

        # Parse safety warnings
        warnings = data.get("safety_warnings", [])
        if isinstance(warnings, list):
            result.safety_warnings = [str(w) for w in warnings if w]

        # Calculate confidence from causes
        if result.probable_causes:
            confidences = [c.get("confidence", 0.5) for c in result.probable_causes]
            result.confidence_score = max(confidences)

        return result

    except json.JSONDecodeError as e:
        result.parse_error = f"JSON parse error: {e!s}"
        return _fallback_parse(response_text, result)


def _fallback_parse(response_text: str, result: ParsedDiagnosisResponse) -> ParsedDiagnosisResponse:
    """
    Fallback parsing when JSON extraction fails.

    Attempts to extract information using regex patterns from
    the raw response text.

    Args:
        response_text: Raw response text.
        result: Partially populated ParsedDiagnosisResponse.

    Returns:
        ParsedDiagnosisResponse with extracted data.
    """
    # Try to extract summary
    summary_match = re.search(r'(?:OSSZEFOGLALO|summary)[:\s]*([^\n]+(?:\n(?![A-Z#*\-\d])[^\n]+)*)',
                             response_text, re.IGNORECASE)
    if summary_match:
        result.summary = summary_match.group(1).strip()

    # Try to extract causes as bullet points
    causes_section = re.search(r'(?:LEHETSEGES OKOK|probable.?causes?)[:\s]*((?:\n[-*\d].*)+)',
                               response_text, re.IGNORECASE)
    if causes_section:
        causes = re.findall(r'[-*\d]+\.?\s*(.+)', causes_section.group(1))
        result.probable_causes = [
            {
                "title": c[:50] + "..." if len(c) > 50 else c,
                "description": c,
                "confidence": 0.5,
                "related_dtc_codes": [],
                "components": [],
            }
            for c in causes[:5]
        ]

    # Try to extract safety warnings
    safety_section = re.search(r'(?:BIZTONSAGI|safety)[:\s]*((?:\n[-*].*)+)',
                               response_text, re.IGNORECASE)
    if safety_section:
        warnings = re.findall(r'[-*]\s*(.+)', safety_section.group(1))
        result.safety_warnings = [w.strip() for w in warnings if w.strip()]

    result.confidence_score = 0.3  # Lower confidence for fallback parsing

    return result


# =============================================================================
# Rule-based Diagnosis Templates
# =============================================================================

DTC_CATEGORY_DESCRIPTIONS = {
    "powertrain": "Hajtaslanc (motor, valto, kipufogo)",
    "body": "Karosszeria es belso rendszerek",
    "chassis": "Futomuro, fek, kormany",
    "network": "Kommunikacio es halozat",
}

SEVERITY_DESCRIPTIONS = {
    "critical": "Kritikus - Azonnal allja meg a jarmut! Biztonsagi kockazat.",
    "high": "Magas - Miniel elobb javittassa meg. Ne halogassa!",
    "medium": "Kozepes - Javitasa szukseges, de nem surgos.",
    "low": "Alacsony - Figyeltesse meg, javitsa alkalom adtan.",
}


def generate_rule_based_diagnosis(
    dtc_codes: list[dict[str, Any]],
    vehicle_info: dict[str, Any],
    recalls: list[dict[str, Any]] | None = None,
    complaints: list[dict[str, Any]] | None = None,
) -> ParsedDiagnosisResponse:
    """
    Generate diagnosis using rule-based logic when LLM is unavailable.

    This function provides a basic diagnosis based on DTC code information
    and database lookups, without requiring an LLM API call.

    Args:
        dtc_codes: List of DTC code information dictionaries.
        vehicle_info: Vehicle information dictionary.
        recalls: Optional NHTSA recalls list.
        complaints: Optional NHTSA complaints list.

    Returns:
        ParsedDiagnosisResponse with rule-based diagnosis.
    """
    result = ParsedDiagnosisResponse()

    # Build summary from DTC codes
    if dtc_codes:
        categories = {d.get("category", "unknown") for d in dtc_codes}
        cat_names = [DTC_CATEGORY_DESCRIPTIONS.get(c, c) for c in categories]
        result.summary = (
            f"A jarmuban {len(dtc_codes)} hibakod talalhato, "
            f"amelyek a kovetkezo rendszereket erintik: {', '.join(cat_names)}. "
            f"Szakszervizben torteno diagnosztika javasolt."
        )
    else:
        result.summary = "Nincs hibakod az adatbazisban. A tunetek alapjan szakszervizben torteno diagnosztika javasolt."

    # Generate probable causes from DTC data
    for dtc in dtc_codes[:5]:
        code = dtc.get("code", "N/A")
        description = dtc.get("description_hu") or dtc.get("description", "")
        severity = dtc.get("severity", "medium")
        possible_causes = dtc.get("possible_causes", [])

        # Create cause entry
        cause_title = f"{code}: {description[:50]}..." if len(description) > 50 else f"{code}: {description}"

        cause = {
            "title": cause_title,
            "description": description,
            "confidence": {"critical": 0.9, "high": 0.75, "medium": 0.6, "low": 0.4}.get(severity, 0.5),
            "related_dtc_codes": [code],
            "components": [],
        }

        # Add specific causes if available
        if possible_causes:
            cause["description"] += "\n\nLehetseges okok:\n" + "\n".join(f"- {c}" for c in possible_causes[:3])

        result.probable_causes.append(cause)

    # Generate diagnostic steps
    result.diagnostic_steps = [
        "1. Olvassa ki a hibakodokat egy OBD-II szkennerrel",
        "2. Ellenorizze a hibakodokhoz tartozo szenzorokat es vezetekeket",
        "3. Vizsgalja meg a kapcsolodo komponenseket (lasd fent)",
        "4. Ha szukseges, vegezzen komponens-specifikus teszteket",
        "5. Javitas utan torolje a hibakodokat es vegezzen tesztmenetet",
    ]

    # Generate repair recommendations from DTC diagnostic steps
    for dtc in dtc_codes[:3]:
        code = dtc.get("code", "")
        diag_steps = dtc.get("diagnostic_steps", [])

        if diag_steps:
            repair = {
                "title": f"Diagnosztika: {code}",
                "description": "\n".join(diag_steps[:3]),
                "estimated_cost_min": 5000,
                "estimated_cost_max": 15000,
                "estimated_cost_currency": "HUF",
                "difficulty": "professional",
                "parts_needed": [],
                "estimated_time_minutes": 30,
            }
            result.recommended_repairs.append(repair)

    # Add recall-based warnings
    if recalls:
        for recall in recalls[:2]:
            component = recall.get("component", "")
            consequence = recall.get("consequence", "")

            if consequence:
                result.safety_warnings.append(
                    f"VISSZAHIVAS ({component}): {consequence[:100]}..."
                )

    # Add critical severity warnings
    for dtc in dtc_codes:
        severity = dtc.get("severity", "medium")
        if severity in ("critical", "high"):
            result.safety_warnings.append(
                SEVERITY_DESCRIPTIONS.get(severity, "")
            )

    # Remove duplicates from warnings
    result.safety_warnings = list(set(result.safety_warnings))

    # Calculate confidence
    if result.probable_causes:
        result.confidence_score = max(c.get("confidence", 0.5) for c in result.probable_causes)
    else:
        result.confidence_score = 0.3

    return result
