"""
Vehicle make normalization - single source of truth.

Canonical NHTSA make spellings used across every layer of the product
(Qdrant filters, PostgreSQL KnownIssue filters, LLM prompts, NHTSA API
calls, diagnosis history). Normalization happens once, at the API schema
boundary (DiagnosisRequest / DiagnosisStreamRequest validators), and the
NHTSA service re-applies it defensively (idempotent no-op for already
canonical input).

Design rules:
- Lookup is case-insensitive (lowercased key).
- Known aliases/typos resolve to the canonical NHTSA spelling
  ("vw" -> "Volkswagen", "chevy" -> "Chevrolet").
- Unknown makes pass through UNCHANGED. No Title-case fallback:
  ``"McLaren".title()`` -> "Mclaren" and ``"RAM".title()`` -> "Ram" both
  return 0 recalls from the case-sensitive NHTSA API, which then gets
  cached. Preserving unknown input restores the safe pass-through
  behavior.
"""

from typing import Dict, FrozenSet

# Lowercase key -> canonical NHTSA make spelling.
# Includes both alias entries (typos, short forms) and identity entries
# for common makes so that any casing of a known brand resolves to the
# exact spelling the NHTSA API expects.
CANONICAL_MAKES: Dict[str, str] = {
    # --- Aliases / typos / short forms ---
    "vw": "Volkswagen",
    "volkswagen ag": "Volkswagen",
    "mercedes": "Mercedes-Benz",
    "mercedes benz": "Mercedes-Benz",
    "mb": "Mercedes-Benz",
    "chevy": "Chevrolet",
    "gm": "General Motors",
    "gmc truck": "GMC",
    "audi ag": "Audi",
    "rolls royce": "Rolls-Royce",
    "mini cooper": "MINI",
    "land-rover": "Land Rover",
    "landrover": "Land Rover",
    "range rover": "Land Rover",
    "rangerover": "Land Rover",
    "alfa-romeo": "Alfa Romeo",
    "porsche ag": "Porsche",
    "bmw ag": "BMW",
    "ds automobiles": "DS",
    # --- Canonical spellings (identity by lowercase key) ---
    "acura": "Acura",
    "alfa romeo": "Alfa Romeo",
    "am general": "AM General",
    "aston martin": "Aston Martin",
    "audi": "Audi",
    "bentley": "Bentley",
    "bmw": "BMW",
    "buick": "Buick",
    "cadillac": "Cadillac",
    "chevrolet": "Chevrolet",
    "chrysler": "Chrysler",
    "citroen": "Citroën",
    "citroën": "Citroën",
    "cupra": "CUPRA",
    "dacia": "Dacia",
    "delorean": "DeLorean",
    "dodge": "Dodge",
    "ds": "DS",
    "fca": "FCA",
    "ferrari": "Ferrari",
    "fiat": "FIAT",
    "ford": "Ford",
    "general motors": "General Motors",
    "genesis": "Genesis",
    "geo": "Geo",
    "gmc": "GMC",
    "honda": "Honda",
    "hummer": "HUMMER",
    "hyundai": "Hyundai",
    "infiniti": "Infiniti",
    "isuzu": "Isuzu",
    "jaguar": "Jaguar",
    "jeep": "Jeep",
    "kia": "Kia",
    "lada": "Lada",
    "lamborghini": "Lamborghini",
    "lancia": "Lancia",
    "land rover": "Land Rover",
    "lexus": "Lexus",
    "lincoln": "Lincoln",
    "lotus": "Lotus",
    "lucid": "Lucid",
    "maserati": "Maserati",
    "mazda": "Mazda",
    "mclaren": "McLaren",
    "mercedes-benz": "Mercedes-Benz",
    "mg": "MG",
    "mercury": "Mercury",
    "mini": "MINI",
    "mitsubishi": "Mitsubishi",
    "nissan": "Nissan",
    "oldsmobile": "Oldsmobile",
    "opel": "Opel",
    "peugeot": "Peugeot",
    "plymouth": "Plymouth",
    "polestar": "Polestar",
    "pontiac": "Pontiac",
    "porsche": "Porsche",
    "ram": "RAM",
    "renault": "Renault",
    "rivian": "Rivian",
    "rolls-royce": "Rolls-Royce",
    "saab": "Saab",
    "saturn": "Saturn",
    "scion": "Scion",
    "seat": "SEAT",
    "skoda": "Skoda",
    "škoda": "Skoda",
    "smart": "smart",
    "subaru": "Subaru",
    "suzuki": "Suzuki",
    "tesla": "Tesla",
    "toyota": "Toyota",
    "vauxhall": "Vauxhall",
    "volkswagen": "Volkswagen",
    "volvo": "Volvo",
}

# Brands not sold in the US — NHTSA recall lookups will always be empty.
# Callers should skip the network round-trip and display an EU-specific
# empty state. Compared against the lowercased *normalized* make.
EU_ONLY_MAKES: FrozenSet[str] = frozenset(
    {
        "skoda",
        "škoda",
        "seat",
        "cupra",
        "opel",
        "vauxhall",
        "peugeot",
        "citroen",
        "citroën",
        "ds",
        "ds automobiles",
        "dacia",
        "lancia",
        "alfa romeo",
        "lada",
        "trabant",
        "wartburg",
        "moskvich",
    }
)


def normalize_make(make: str) -> str:
    """Return the canonical NHTSA spelling for a vehicle make.

    Strips surrounding whitespace and resolves known aliases/casings via
    :data:`CANONICAL_MAKES`. Unknown makes are returned unchanged (after
    stripping) — NO Title-case fallback, so brands like "McLaren", "RAM"
    or "AM General" are never corrupted.

    Idempotent: ``normalize_make(normalize_make(x)) == normalize_make(x)``.
    """
    cleaned = make.strip()
    if not cleaned:
        return cleaned
    return CANONICAL_MAKES.get(cleaned.lower(), cleaned)


def is_eu_only(make: str) -> bool:
    """Return True if the make is an EU-only brand with no NHTSA coverage."""
    return normalize_make(make).lower() in EU_ONLY_MAKES
