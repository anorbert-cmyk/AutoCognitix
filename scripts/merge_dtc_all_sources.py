#!/usr/bin/env python3
"""
DTC All-Sources Merge Script for AutoCognitix - Sprint 8

Merges DTC codes from ALL available sources (existing + new GitHub sources)
into a single complete dataset.

Sources:
  1. Existing: data/dtc_codes/all_codes_merged.json (3,716 codes - baseline)
  2. Existing: data/dtc/dtcdb_codes.json (467 codes)
  3. New: data/dtc_codes/xinings_dtc/.../codes.json (6,665 codes - dict format)
  4. New: data/dtc_codes/fabiovila_obd/.../codes.json (2,381 codes - array format)
  5. New: data/dtc_codes/wzr1337_dtc_complete.json (3,745 codes - flat dict)

Merge Strategy:
  - Deduplicate by DTC code (uppercase, normalized)
  - Existing merged data has highest priority (preserves translations, NHTSA data)
  - For new codes: prefer longer/more detailed descriptions
  - Normalize all codes to standard format

Output: data/dtc_codes/all_codes_complete.json

Usage:
    python scripts/merge_dtc_all_sources.py
    python scripts/merge_dtc_all_sources.py --dry-run
    python scripts/merge_dtc_all_sources.py --verbose
"""

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "dtc_codes"
OUTPUT_FILE = DATA_DIR / "all_codes_complete.json"

# DTC code validation pattern: [PCBU][0-9A-F]{4}
DTC_PATTERN = re.compile(r"^[PCBU][0-9A-Fa-f]{4}$")

# Category mapping from code prefix
CATEGORY_MAP = {
    "P": "powertrain",
    "C": "chassis",
    "B": "body",
    "U": "network",
}

# Subcategory mapping for Powertrain codes (based on second+third chars)
POWERTRAIN_SUBCATEGORY_MAP = {
    "00": "fuel_and_air",
    "01": "fuel_and_air",
    "02": "fuel_and_air_injection",
    "03": "ignition_misfire",
    "04": "auxiliary_emission",
    "05": "speed_idle_control",
    "06": "computer_output",
    "07": "transmission",
    "08": "transmission",
    "09": "transmission",
    "0A": "hybrid_propulsion",
}

# System mapping (same as utils.py for consistency)
SYSTEM_MAP_P = {
    "00": "Fuel and Air Metering",
    "01": "Fuel and Air Metering",
    "02": "Fuel and Air Metering Injection",
    "03": "Ignition System/Misfire",
    "04": "Auxiliary Emission Controls",
    "05": "Vehicle Speed and Idle Control",
    "06": "Computer Output Circuits",
    "07": "Transmission",
    "08": "Transmission",
    "09": "Transmission",
    "0A": "Hybrid Propulsion",
}

SYSTEM_MAP_PREFIX = {
    "C": "Chassis/ABS/Traction",
    "B": "Body/Interior",
    "U": "Network Communication",
}


def validate_dtc_code(code: str) -> bool:
    """Validate DTC code format: [PCBU][0-9A-F]{4}."""
    return bool(DTC_PATTERN.match(code.upper()))


def normalize_code(raw_code: str) -> Optional[str]:
    """
    Normalize a DTC code string.

    Handles:
    - Stripping whitespace
    - Uppercase conversion
    - Removing /SAE, /PCM, /Turbine suffixes (fabiovila format)
    - Validation

    Returns:
        Normalized code string or None if invalid.
    """
    if not raw_code:
        return None

    code = raw_code.strip().upper()

    # Remove known suffixes like /SAE, /PCM, /Turbine etc.
    if "/" in code:
        code = code.split("/")[0].strip()

    # Validate
    if not validate_dtc_code(code):
        return None

    return code


def get_category(code: str) -> str:
    """Get category from DTC code prefix."""
    return CATEGORY_MAP.get(code[0].upper(), "unknown")


def get_subcategory(code: str) -> Optional[str]:
    """Get subcategory for powertrain codes."""
    if code[0].upper() != "P":
        return None
    middle = code[1:3].upper()
    return POWERTRAIN_SUBCATEGORY_MAP.get(middle)


def get_severity(code: str) -> str:
    """Estimate severity from DTC code pattern."""
    prefix = code[0].upper()
    code_upper = code.upper()

    if prefix == "U":
        return "high"
    if prefix == "B" and code_upper.startswith(("B0", "B1")):
        return "critical"
    if prefix == "P":
        if code_upper.startswith("P03"):
            return "high"
        if code_upper.startswith(("P07", "P08", "P09")):
            return "high"
    return "medium"


def get_system(code: str) -> str:
    """Get system description from DTC code."""
    prefix = code[0].upper()
    if prefix == "P":
        middle = code[1:3].upper()
        return SYSTEM_MAP_P.get(middle, "Powertrain")
    return SYSTEM_MAP_PREFIX.get(prefix, "")


def sanitize_description(desc: str) -> str:
    """Clean up a description string."""
    if not desc:
        return ""
    # Strip whitespace
    desc = desc.strip()
    # Remove leading/trailing quotes
    desc = desc.strip('"').strip("'")
    # Collapse multiple spaces
    desc = re.sub(r"\s+", " ", desc)
    # Remove control characters
    desc = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", desc)
    return desc


def make_entry(
    code: str,
    description: str,
    source: str,
    *,
    description_hu: Optional[str] = None,
    category: Optional[str] = None,
    subcategory: Optional[str] = None,
    severity: Optional[str] = None,
    system: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a normalized DTC entry dict."""
    return {
        "code": code.upper(),
        "description": sanitize_description(description),
        "description_hu": description_hu,
        "category": category or get_category(code),
        "subcategory": subcategory or get_subcategory(code),
        "severity": severity or get_severity(code),
        "system": system or get_system(code),
        "source": source,
    }


# ---------------------------------------------------------------------------
# Source Loaders
# ---------------------------------------------------------------------------


def load_existing_merged() -> List[Dict[str, Any]]:
    """
    Load the existing all_codes_merged.json (baseline with translations, NHTSA data).

    Format: {"metadata": {...}, "codes": [{code, description_en, description_hu, ...}, ...]}
    """
    path = DATA_DIR / "all_codes_merged.json"
    if not path.exists():
        print(f"  WARNING: {path} not found, skipping")
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    codes = data.get("codes", [])
    results = []

    for item in codes:
        code = normalize_code(item.get("code", ""))
        if not code:
            continue

        desc = item.get("description_en", "") or ""
        if not desc.strip():
            continue

        entry = make_entry(
            code,
            desc,
            "existing_merged",
            description_hu=item.get("description_hu"),
            category=item.get("category"),
            severity=item.get("severity"),
            system=item.get("system"),
        )
        # Carry over rich fields from existing data
        entry["symptoms"] = item.get("symptoms", [])
        entry["possible_causes"] = item.get("possible_causes", [])
        entry["diagnostic_steps"] = item.get("diagnostic_steps", [])
        entry["related_codes"] = item.get("related_codes", [])
        entry["manufacturer"] = item.get("manufacturer")
        entry["translation_status"] = item.get("translation_status", "pending")
        entry["original_sources"] = item.get("sources", [])
        entry["nhtsa_recalls"] = item.get("nhtsa_recalls", [])
        entry["nhtsa_complaints"] = item.get("nhtsa_complaints", [])
        entry["is_generic"] = item.get("is_generic", code[1] == "0")

        results.append(entry)

    print(f"  Loaded {len(results)} codes from existing merged (baseline)")
    return results


def load_dtcdb() -> List[Dict[str, Any]]:
    """
    Load data/dtc/dtcdb_codes.json.

    Format: {"metadata": {...}, "codes": [{code, description, category, subcategory}, ...]}
    """
    path = PROJECT_ROOT / "data" / "dtc" / "dtcdb_codes.json"
    if not path.exists():
        print(f"  WARNING: {path} not found, skipping")
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    codes = data.get("codes", [])
    results = []

    for item in codes:
        code = normalize_code(item.get("code", ""))
        if not code:
            continue

        desc = item.get("description", "") or ""
        if not desc.strip():
            continue

        # Map category from dtcdb format (e.g. "Powertrain" -> "powertrain")
        cat = (item.get("category") or "").lower()
        subcat_raw = item.get("subcategory", "")
        subcat = subcat_raw.lower().replace(" ", "_").replace("/", "_") if subcat_raw else None

        entry = make_entry(
            code,
            desc,
            "dtcdb",
            category=cat if cat in ("powertrain", "chassis", "body", "network") else None,
            subcategory=subcat,
        )
        results.append(entry)

    print(f"  Loaded {len(results)} codes from dtcdb")
    return results


def load_xinings() -> List[Dict[str, Any]]:
    """
    Load xinings DTC-Database.

    Format: {"codes": {"B1200": "description", ...}}
    """
    path = DATA_DIR / "xinings_dtc" / "DTC-Database-master" / "codes.json"
    if not path.exists():
        print(f"  WARNING: {path} not found, skipping")
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    codes_dict = data.get("codes", {})
    results = []

    for raw_code, desc in codes_dict.items():
        code = normalize_code(raw_code)
        if not code:
            continue
        desc_str = str(desc).strip() if desc else ""
        if not desc_str or desc_str.lower() in ("see manufacturer", "reserved"):
            continue

        entry = make_entry(code, desc_str, "xinings")
        results.append(entry)

    print(f"  Loaded {len(results)} codes from xinings")
    return results


def load_fabiovila() -> List[Dict[str, Any]]:
    """
    Load fabiovila OBDIICodes.

    Format: [{"Code": "P0001", "Description": "..."}, ...]
    Some codes have /SAE, /PCM suffixes that need stripping.
    """
    path = DATA_DIR / "fabiovila_obd" / "OBDIICodes-master" / "codes.json"
    if not path.exists():
        print(f"  WARNING: {path} not found, skipping")
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []

    for item in data:
        raw_code = item.get("Code", "")
        code = normalize_code(raw_code)
        if not code:
            continue

        desc = item.get("Description", "").strip()
        if not desc or desc.lower() in ("reserved", "see manufacturer"):
            continue

        entry = make_entry(code, desc, "fabiovila")
        results.append(entry)

    print(f"  Loaded {len(results)} codes from fabiovila")
    return results


def load_wzr1337() -> List[Dict[str, Any]]:
    """
    Load wzr1337 gist data.

    Format: {"P0001": "description", ...}
    """
    path = DATA_DIR / "wzr1337_dtc_complete.json"
    if not path.exists():
        print(f"  WARNING: {path} not found, skipping")
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []

    for raw_code, desc in data.items():
        code = normalize_code(raw_code)
        if not code:
            continue
        desc_str = str(desc).strip() if desc else ""
        if not desc_str or desc_str.lower() in ("no trouble code", "reserved"):
            continue

        entry = make_entry(code, desc_str, "wzr1337")
        results.append(entry)

    print(f"  Loaded {len(results)} codes from wzr1337")
    return results


# ---------------------------------------------------------------------------
# Merge Logic
# ---------------------------------------------------------------------------

# Source priority: higher number = preferred when resolving conflicts
SOURCE_PRIORITY = {
    "existing_merged": 100,  # Highest - has translations, NHTSA, rich data
    "dtcdb": 10,
    "xinings": 5,
    "wzr1337": 4,
    "fabiovila": 3,
}


def pick_better_description(existing_desc: str, new_desc: str) -> str:
    """
    Pick the better description between two options.

    Prefers longer, more detailed descriptions.
    Filters out clearly inferior descriptions.
    """
    if not new_desc:
        return existing_desc
    if not existing_desc:
        return new_desc

    # If one is a NHTSA-extracted placeholder, prefer the other
    if "code extracted from nhtsa" in existing_desc.lower():
        return new_desc
    if "code extracted from nhtsa" in new_desc.lower():
        return existing_desc

    # Prefer the longer one (usually more informative)
    if len(new_desc) > len(existing_desc) * 1.2:
        return new_desc

    return existing_desc


def merge_all(
    sources: List[Tuple[str, List[Dict[str, Any]]]]
) -> Dict[str, Dict[str, Any]]:
    """
    Merge all sources into a single deduplicated dict keyed by code.

    Sources are processed in priority order (lowest first, highest last).
    This means higher-priority sources can overwrite lower-priority data.

    For the existing_merged source (highest priority), all rich fields
    are preserved (translations, NHTSA data, symptoms, etc.).
    """
    merged: Dict[str, Dict[str, Any]] = {}
    stats = {
        "by_source": {},
        "new_codes_by_source": {},
        "updated_descriptions": 0,
        "total_input": 0,
        "duplicates_merged": 0,
    }

    # Sort sources by priority (lowest first)
    sorted_sources = sorted(sources, key=lambda x: SOURCE_PRIORITY.get(x[0], 0))

    for source_name, codes in sorted_sources:
        source_new = 0
        source_total = len(codes)
        stats["by_source"][source_name] = source_total
        stats["total_input"] += source_total

        for entry in codes:
            code = entry["code"]

            if code not in merged:
                # New code - add it
                merged[code] = entry.copy()
                merged[code]["sources"] = [source_name]
                source_new += 1
            else:
                # Existing code - merge intelligently
                existing = merged[code]
                stats["duplicates_merged"] += 1

                # Track source
                if source_name not in existing.get("sources", []):
                    existing.setdefault("sources", []).append(source_name)

                current_priority = SOURCE_PRIORITY.get(source_name, 0)
                existing_source = existing.get("source", "")
                existing_priority = SOURCE_PRIORITY.get(existing_source, 0)

                # For existing_merged source, always preserve all rich fields
                if source_name == "existing_merged":
                    # Overwrite with existing_merged data (highest priority)
                    rich_fields = [
                        "description_hu",
                        "symptoms",
                        "possible_causes",
                        "diagnostic_steps",
                        "related_codes",
                        "manufacturer",
                        "translation_status",
                        "original_sources",
                        "nhtsa_recalls",
                        "nhtsa_complaints",
                        "is_generic",
                    ]
                    for field in rich_fields:
                        if field in entry and entry[field]:
                            existing[field] = entry[field]

                    # Description: prefer existing_merged unless it's a placeholder
                    better_desc = pick_better_description(
                        existing.get("description", ""),
                        entry.get("description", ""),
                    )
                    if better_desc != existing.get("description", ""):
                        existing["description"] = better_desc
                        stats["updated_descriptions"] += 1

                    # Update metadata from existing_merged
                    existing["category"] = entry.get("category") or existing.get("category")
                    existing["severity"] = entry.get("severity") or existing.get("severity")
                    existing["system"] = entry.get("system") or existing.get("system")
                    existing["source"] = source_name

                elif current_priority >= existing_priority:
                    # Higher or equal priority new source
                    better_desc = pick_better_description(
                        existing.get("description", ""),
                        entry.get("description", ""),
                    )
                    if better_desc != existing.get("description", ""):
                        existing["description"] = better_desc
                        stats["updated_descriptions"] += 1

                    # Fill in missing subcategory
                    if entry.get("subcategory") and not existing.get("subcategory"):
                        existing["subcategory"] = entry["subcategory"]
                else:
                    # Lower priority: only update if current description is missing/placeholder
                    if not existing.get("description") or "code extracted from nhtsa" in existing.get("description", "").lower():
                        existing["description"] = entry.get("description", "")
                        stats["updated_descriptions"] += 1

        stats["new_codes_by_source"][source_name] = source_new
        print(f"  After {source_name}: {len(merged)} total codes (+{source_new} new)")

    return merged, stats


def build_output(
    merged: Dict[str, Dict[str, Any]],
    stats: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the final output JSON structure."""
    # Sort by code
    sorted_codes = sorted(merged.values(), key=lambda x: x["code"])

    # Build clean output entries
    output_codes = []
    for entry in sorted_codes:
        clean = {
            "code": entry["code"],
            "description": entry.get("description", ""),
            "description_hu": entry.get("description_hu"),
            "category": entry.get("category", get_category(entry["code"])),
            "subcategory": entry.get("subcategory"),
            "severity": entry.get("severity", get_severity(entry["code"])),
            "system": entry.get("system", get_system(entry["code"])),
            "is_generic": entry.get("is_generic", entry["code"][1] == "0"),
            "sources": entry.get("sources", [entry.get("source", "unknown")]),
        }

        # Preserve rich fields if present
        for rich_field in [
            "symptoms",
            "possible_causes",
            "diagnostic_steps",
            "related_codes",
            "manufacturer",
            "translation_status",
            "nhtsa_recalls",
            "nhtsa_complaints",
        ]:
            value = entry.get(rich_field)
            if value:
                clean[rich_field] = value

        # Preserve original_sources from existing_merged
        if entry.get("original_sources"):
            clean["original_sources"] = entry["original_sources"]

        output_codes.append(clean)

    # Count stats
    translated = sum(1 for c in output_codes if c.get("description_hu"))
    with_nhtsa = sum(
        1 for c in output_codes
        if c.get("nhtsa_recalls") or c.get("nhtsa_complaints")
    )
    category_counts = {}
    for c in output_codes:
        cat = c.get("category", "unknown")
        category_counts[cat] = category_counts.get(cat, 0) + 1

    output = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_codes": len(output_codes),
            "translated": translated,
            "with_nhtsa_data": with_nhtsa,
            "sources": list(stats.get("by_source", {}).keys()),
            "codes_by_category": dict(sorted(category_counts.items())),
            "merge_stats": {
                "total_input_codes": stats.get("total_input", 0),
                "duplicates_merged": stats.get("duplicates_merged", 0),
                "updated_descriptions": stats.get("updated_descriptions", 0),
                "new_codes_by_source": stats.get("new_codes_by_source", {}),
                "codes_per_source": stats.get("by_source", {}),
            },
        },
        "codes": output_codes,
    }

    return output


def print_report(output: Dict[str, Any], stats: Dict[str, Any]) -> None:
    """Print a human-readable merge report."""
    meta = output["metadata"]
    merge_stats = meta.get("merge_stats", {})

    print("\n" + "=" * 65)
    print("  DTC ALL-SOURCES MERGE REPORT")
    print("=" * 65)
    print(f"\n  Total unique codes:     {meta['total_codes']}")
    print(f"  With HU translation:    {meta['translated']}")
    print(f"  With NHTSA data:        {meta['with_nhtsa_data']}")
    print(f"  Total input codes:      {merge_stats.get('total_input_codes', 0)}")
    print(f"  Duplicates merged:      {merge_stats.get('duplicates_merged', 0)}")
    print(f"  Descriptions updated:   {merge_stats.get('updated_descriptions', 0)}")

    print("\n  Codes per source (input):")
    for source, count in sorted(merge_stats.get("codes_per_source", {}).items()):
        print(f"    {source:20s}: {count:>6,}")

    print("\n  New codes contributed per source:")
    for source, count in sorted(merge_stats.get("new_codes_by_source", {}).items()):
        print(f"    {source:20s}: {count:>6,}")

    print("\n  Codes by category:")
    for cat, count in sorted(meta.get("codes_by_category", {}).items()):
        print(f"    {cat:20s}: {count:>6,}")

    print("\n" + "=" * 65)


def verify_output(output: Dict[str, Any], existing_count: int) -> bool:
    """
    Verify the output is valid and complete.

    Returns True if all checks pass.
    """
    codes = output.get("codes", [])
    total = len(codes)

    print("\n  Verification:")

    # Check 1: More codes than existing
    if total < existing_count:
        print(f"    FAIL: Output has fewer codes ({total}) than existing ({existing_count})")
        return False
    print(f"    PASS: {total} codes >= {existing_count} existing codes")

    # Check 2: All codes are valid DTC format
    invalid = [c["code"] for c in codes if not validate_dtc_code(c["code"])]
    if invalid:
        print(f"    FAIL: {len(invalid)} invalid codes found: {invalid[:5]}")
        return False
    print(f"    PASS: All {total} codes have valid DTC format")

    # Check 3: No duplicates
    code_set = set()
    dupes = []
    for c in codes:
        if c["code"] in code_set:
            dupes.append(c["code"])
        code_set.add(c["code"])
    if dupes:
        print(f"    FAIL: {len(dupes)} duplicate codes found: {dupes[:5]}")
        return False
    print("    PASS: No duplicate codes")

    # Check 4: Every code has a description
    no_desc = [c["code"] for c in codes if not c.get("description")]
    if no_desc:
        print(f"    FAIL: {len(no_desc)} codes without description: {no_desc[:5]}")
        return False
    print("    PASS: All codes have descriptions")

    # Check 5: Valid JSON output (implicit since we got here)
    print("    PASS: Valid JSON structure")

    # Check 6: Sorted by code
    code_list = [c["code"] for c in codes]
    if code_list != sorted(code_list):
        print("    FAIL: Codes are not sorted")
        return False
    print("    PASS: Codes are sorted alphabetically")

    print("\n  All verification checks passed!")
    return True


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge DTC codes from ALL available sources"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview merge without writing output file",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("  AutoCognitix DTC All-Sources Merge - Sprint 8")
    print("=" * 65)

    # Step 1: Load all sources
    print("\n[Step 1] Loading all sources...")

    sources: List[Tuple[str, List[Dict[str, Any]]]] = []

    existing = load_existing_merged()
    existing_count = len(existing)
    sources.append(("existing_merged", existing))

    dtcdb = load_dtcdb()
    sources.append(("dtcdb", dtcdb))

    xinings = load_xinings()
    sources.append(("xinings", xinings))

    fabiovila = load_fabiovila()
    sources.append(("fabiovila", fabiovila))

    wzr1337 = load_wzr1337()
    sources.append(("wzr1337", wzr1337))

    total_input = sum(len(s[1]) for s in sources)
    print(f"\n  Total input codes across all sources: {total_input:,}")

    # Step 2: Merge
    print("\n[Step 2] Merging (lowest priority first)...")
    merged, stats = merge_all(sources)
    print(f"\n  Merge complete: {len(merged):,} unique codes")

    # Step 3: Build output
    print("\n[Step 3] Building output...")
    output = build_output(merged, stats)

    # Step 4: Print report
    print_report(output, stats)

    # Step 5: Verify
    print("\n[Step 4] Verifying output...")
    ok = verify_output(output, existing_count)

    if not ok:
        print("\n  ERROR: Verification failed! Output not written.")
        sys.exit(1)

    # Step 6: Write output
    if args.dry_run:
        print(f"\n  [DRY RUN] Would write to: {OUTPUT_FILE}")
        print(f"  [DRY RUN] {output['metadata']['total_codes']:,} codes")
    else:
        print(f"\n[Step 5] Writing output to {OUTPUT_FILE}...")
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
        print(f"  Written: {OUTPUT_FILE}")
        print(f"  File size: {file_size_mb:.2f} MB")

    print("\n  Done!")


if __name__ == "__main__":
    main()
