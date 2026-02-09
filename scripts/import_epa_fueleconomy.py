#!/usr/bin/env python3
"""
EPA FuelEconomy.gov Data Importer

Downloads and processes the EPA vehicles.csv dataset (1984-2026, ~49,500+ records).
Filters to 2000+ model years, normalizes field names, and outputs structured JSON files.

Data source: https://fueleconomy.gov/feg/epadata/vehicles.csv

Output files:
  - data/epa/vehicles_complete.json   - All 2000+ vehicles (streaming write)
  - data/epa/vehicles_by_year/*.json  - One file per model year
  - data/epa/fuel_types_summary.json  - Statistics by fuel type
  - data/epa/engine_specs.json        - Unique engine configurations
  - data/epa/import_stats.json        - Full import statistics

Usage:
  python scripts/import_epa_fueleconomy.py [--min-year 2000] [--csv-path data/epa/vehicles.csv]
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "epa"
CSV_PATH = DATA_DIR / "vehicles.csv"

# Minimum year filter (default: 2000)
MIN_YEAR = 2000


def safe_int(value, default=0):
    """Convert value to int, returning default on failure."""
    if not value or not str(value).strip():
        return default
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default


def safe_float(value, default=0.0):
    """Convert value to float, returning default on failure."""
    if not value or not str(value).strip():
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def classify_fuel_type(fuel_type, atv_type, ev_motor, phev_blended):
    """Classify a vehicle into a broad fuel category."""
    ft_lower = fuel_type.lower() if fuel_type else ""
    atv_lower = atv_type.lower() if atv_type else ""

    if atv_lower == "ev" or ft_lower == "electricity":
        return "Electric (EV)"
    if phev_blended == "true" or "and electricity" in ft_lower:
        return "Plug-in Hybrid (PHEV)"
    if "hybrid" in atv_lower:
        return "Hybrid"
    if "e85" in ft_lower:
        return "Flex-Fuel (E85)"
    if "diesel" in ft_lower:
        return "Diesel"
    if "hydrogen" in ft_lower:
        return "Hydrogen Fuel Cell"
    if "cng" in ft_lower or "natural gas" in ft_lower:
        return "Natural Gas (CNG)"
    if "propane" in ft_lower:
        return "Propane"
    if "premium" in ft_lower:
        return "Gasoline (Premium)"
    if "midgrade" in ft_lower:
        return "Gasoline (Midgrade)"
    return "Gasoline (Regular)"


def build_engine_description(row):
    """Build a human-readable engine description string."""
    parts = []
    displ = safe_float(row.get("displ"))
    cyl = safe_int(row.get("cylinders"))
    eng_dscr = (row.get("eng_dscr") or "").strip()
    t_charger = (row.get("tCharger") or "").strip()
    s_charger = (row.get("sCharger") or "").strip()
    ev_motor = (row.get("evMotor") or "").strip()

    if ev_motor:
        return ev_motor

    if displ > 0:
        parts.append(f"{displ}L")
    if cyl > 0:
        parts.append(f"{cyl}-cylinder")
    if t_charger:
        parts.append("Turbo")
    if s_charger:
        parts.append("Supercharged")
    if eng_dscr:
        # Clean up engine description - remove parentheses and extra whitespace
        clean_desc = eng_dscr.replace("(", "").replace(")", "").strip()
        # Only add if it provides meaningful info not already captured
        if clean_desc and clean_desc not in ("FFS", "SFI", "MFI", "GUZZLER"):
            parts.append(f"({clean_desc})")

    return " ".join(parts) if parts else "N/A"


def normalize_record(row):
    """Normalize a CSV row into our standard schema."""
    year = safe_int(row.get("year"))
    fuel_type_raw = (row.get("fuelType") or "").strip()
    atv_type = (row.get("atvType") or "").strip()
    ev_motor = (row.get("evMotor") or "").strip()
    phev_blended = (row.get("phevBlended") or "").strip().lower()

    record = {
        "id": safe_int(row.get("id")),
        "make": (row.get("make") or "").strip(),
        "model": (row.get("model") or "").strip(),
        "base_model": (row.get("baseModel") or "").strip(),
        "model_year": year,
        "vehicle_class": (row.get("VClass") or "").strip(),
        "drive_type": (row.get("drive") or "").strip(),
        "transmission": (row.get("trany") or "").strip(),
        "cylinders": safe_int(row.get("cylinders")),
        "displacement_liters": safe_float(row.get("displ")),
        "engine_description": build_engine_description(row),
        "fuel_type": fuel_type_raw,
        "fuel_type_primary": (row.get("fuelType1") or "").strip(),
        "fuel_type_secondary": (row.get("fuelType2") or "").strip(),
        "fuel_category": classify_fuel_type(
            fuel_type_raw, atv_type, ev_motor, phev_blended
        ),
        "mpg_city": safe_int(row.get("city08")),
        "mpg_highway": safe_int(row.get("highway08")),
        "mpg_combined": safe_int(row.get("comb08")),
        "co2_grams_per_mile": safe_float(row.get("co2TailpipeGpm")),
        "annual_fuel_cost": safe_int(row.get("fuelCost08")),
        "annual_barrels": safe_float(row.get("barrels08")),
        "you_save_spend": safe_int(row.get("youSaveSpend")),
        "fe_score": safe_int(row.get("feScore")),
        "ghg_score": safe_int(row.get("ghgScore")),
        # EV/PHEV specific
        "atv_type": atv_type,
        "ev_motor": ev_motor,
        "phev_blended": phev_blended == "true",
        "range_miles": safe_int(row.get("range")),
        "range_city_miles": safe_float(row.get("rangeCity")),
        "range_highway_miles": safe_float(row.get("rangeHwy")),
        "charge_time_120v_hours": safe_float(row.get("charge120")),
        "charge_time_240v_hours": safe_float(row.get("charge240")),
        "start_stop": (row.get("startStop") or "").strip(),
        # Guzzler flag
        "is_guzzler": bool((row.get("guzzler") or "").strip()),
        # Turbo/Supercharger flags
        "has_turbo": bool((row.get("tCharger") or "").strip()),
        "has_supercharger": bool((row.get("sCharger") or "").strip()),
    }
    return record


def process_csv(csv_path, min_year):
    """
    Process the EPA CSV file and generate all output files.

    Uses streaming approach: writes vehicles_complete.json incrementally
    to avoid loading all records into memory at once.
    """
    print(f"Reading CSV: {csv_path}")
    print(f"Filtering to year >= {min_year}")

    # Ensure output directories exist
    by_year_dir = DATA_DIR / "vehicles_by_year"
    by_year_dir.mkdir(parents=True, exist_ok=True)

    # Accumulators
    vehicles_by_year = defaultdict(list)
    fuel_type_stats = defaultdict(lambda: {
        "count": 0,
        "makes": set(),
        "years": set(),
        "avg_mpg_combined": [],
        "avg_co2": [],
    })
    engine_specs_map = {}  # key: (make, model, year, engine_desc, transmission, drive)

    total_csv = 0
    total_filtered = 0
    skipped_before_min_year = 0
    makes = set()
    models = set()  # (make, model) pairs
    ev_count = 0
    phev_count = 0
    hybrid_count = 0
    conventional_count = 0

    # Streaming write for vehicles_complete.json
    complete_path = DATA_DIR / "vehicles_complete.json"
    with open(complete_path, "w", encoding="utf-8") as complete_file:
        complete_file.write("[\n")
        first_record = True

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                total_csv += 1
                year = safe_int(row.get("year"))

                if year < min_year:
                    skipped_before_min_year += 1
                    continue

                record = normalize_record(row)
                total_filtered += 1

                # Write to complete file (streaming)
                if not first_record:
                    complete_file.write(",\n")
                json.dump(record, complete_file, ensure_ascii=False)
                first_record = False

                # Accumulate by year
                vehicles_by_year[year].append(record)

                # Stats
                makes.add(record["make"])
                models.add((record["make"], record["model"]))

                cat = record["fuel_category"]
                if cat == "Electric (EV)":
                    ev_count += 1
                elif cat == "Plug-in Hybrid (PHEV)":
                    phev_count += 1
                elif cat == "Hybrid":
                    hybrid_count += 1
                else:
                    conventional_count += 1

                # Fuel type statistics
                ft = record["fuel_type"] or "Unknown"
                ft_stats = fuel_type_stats[ft]
                ft_stats["count"] += 1
                ft_stats["makes"].add(record["make"])
                ft_stats["years"].add(year)
                if record["mpg_combined"] > 0:
                    ft_stats["avg_mpg_combined"].append(record["mpg_combined"])
                if record["co2_grams_per_mile"] > 0:
                    ft_stats["avg_co2"].append(record["co2_grams_per_mile"])

                # Engine specs (unique configurations)
                spec_key = (
                    record["make"],
                    record["model"],
                    record["model_year"],
                    record["engine_description"],
                    record["transmission"],
                    record["drive_type"],
                )
                if spec_key not in engine_specs_map:
                    engine_specs_map[spec_key] = {
                        "make": record["make"],
                        "model": record["model"],
                        "year": record["model_year"],
                        "engine": record["engine_description"],
                        "cylinders": record["cylinders"],
                        "displacement": record["displacement_liters"],
                        "fuel_type": record["fuel_type"],
                        "fuel_category": record["fuel_category"],
                        "transmission": record["transmission"],
                        "drive": record["drive_type"],
                        "mpg_combined": record["mpg_combined"],
                        "co2_grams_per_mile": record["co2_grams_per_mile"],
                        "ev_motor": record["ev_motor"] or None,
                        "has_turbo": record["has_turbo"],
                        "has_supercharger": record["has_supercharger"],
                    }

                # Progress indicator
                if total_csv % 10000 == 0:
                    print(f"  Processed {total_csv} rows...")

        complete_file.write("\n]\n")

    print(f"  Total CSV rows: {total_csv}")
    print(f"  Filtered (>= {min_year}): {total_filtered}")
    print(f"  Skipped (< {min_year}): {skipped_before_min_year}")

    # Write vehicles by year
    print(f"\nWriting vehicles by year ({len(vehicles_by_year)} years)...")
    for year, records in sorted(vehicles_by_year.items()):
        year_path = by_year_dir / f"{year}.json"
        with open(year_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"  {year}: {len(records)} vehicles")

    # Write fuel types summary
    print("\nWriting fuel types summary...")
    fuel_summary = {}
    for ft, stats in sorted(fuel_type_stats.items(), key=lambda x: -x[1]["count"]):
        avg_mpg = (
            round(sum(stats["avg_mpg_combined"]) / len(stats["avg_mpg_combined"]), 1)
            if stats["avg_mpg_combined"]
            else 0
        )
        avg_co2 = (
            round(sum(stats["avg_co2"]) / len(stats["avg_co2"]), 1)
            if stats["avg_co2"]
            else 0
        )
        fuel_summary[ft] = {
            "count": stats["count"],
            "unique_makes": len(stats["makes"]),
            "year_range": f"{min(stats['years'])}-{max(stats['years'])}",
            "avg_mpg_combined": avg_mpg,
            "avg_co2_grams_per_mile": avg_co2,
        }

    fuel_summary_path = DATA_DIR / "fuel_types_summary.json"
    with open(fuel_summary_path, "w", encoding="utf-8") as f:
        json.dump(fuel_summary, f, ensure_ascii=False, indent=2)

    # Write engine specs
    print(f"\nWriting engine specs ({len(engine_specs_map)} unique configurations)...")
    engine_specs = list(engine_specs_map.values())
    # Sort by make, model, year
    engine_specs.sort(key=lambda x: (x["make"], x["model"], x["year"]))

    engine_specs_path = DATA_DIR / "engine_specs.json"
    # Streaming write for large file
    with open(engine_specs_path, "w", encoding="utf-8") as f:
        f.write("[\n")
        for i, spec in enumerate(engine_specs):
            if i > 0:
                f.write(",\n")
            json.dump(spec, f, ensure_ascii=False)
        f.write("\n]\n")

    # Write import stats
    stats = {
        "source": "EPA FuelEconomy.gov",
        "source_url": "https://fueleconomy.gov/feg/epadata/vehicles.csv",
        "import_date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "csv_total_records": total_csv,
        "filtered_records": total_filtered,
        "min_year_filter": min_year,
        "skipped_records": skipped_before_min_year,
        "year_range": f"{min(vehicles_by_year.keys())}-{max(vehicles_by_year.keys())}",
        "unique_makes": len(makes),
        "unique_models": len(models),
        "makes_list": sorted(makes),
        "vehicle_counts_by_category": {
            "electric_ev": ev_count,
            "plug_in_hybrid_phev": phev_count,
            "hybrid": hybrid_count,
            "conventional": conventional_count,
        },
        "unique_engine_configurations": len(engine_specs_map),
        "output_files": {
            "vehicles_complete": str(complete_path.relative_to(PROJECT_ROOT)),
            "vehicles_by_year": str(by_year_dir.relative_to(PROJECT_ROOT)),
            "fuel_types_summary": str(fuel_summary_path.relative_to(PROJECT_ROOT)),
            "engine_specs": str(engine_specs_path.relative_to(PROJECT_ROOT)),
        },
    }

    stats_path = DATA_DIR / "import_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("EPA FuelEconomy.gov Import Complete")
    print("=" * 60)
    print(f"Total CSV records:           {total_csv:,}")
    print(f"Filtered records (>={min_year}):  {total_filtered:,}")
    print(f"Year range:                  {stats['year_range']}")
    print(f"Unique makes:                {len(makes)}")
    print(f"Unique make/model combos:    {len(models):,}")
    print(f"Unique engine configs:       {len(engine_specs_map):,}")
    print("\nVehicle Categories:")
    print(f"  Electric (EV):             {ev_count:,}")
    print(f"  Plug-in Hybrid (PHEV):     {phev_count:,}")
    print(f"  Hybrid:                    {hybrid_count:,}")
    print(f"  Conventional:              {conventional_count:,}")
    print("\nOutput files:")
    print(f"  {complete_path}")
    print(f"  {by_year_dir}/ ({len(vehicles_by_year)} files)")
    print(f"  {fuel_summary_path}")
    print(f"  {engine_specs_path}")
    print(f"  {stats_path}")
    print("=" * 60)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Import EPA FuelEconomy.gov vehicle data"
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=MIN_YEAR,
        help=f"Minimum model year to include (default: {MIN_YEAR})",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=str(CSV_PATH),
        help=f"Path to vehicles.csv (default: {CSV_PATH})",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"ERROR: CSV file not found at {csv_path}")
        print("Download it first:")
        print(
            '  curl -L "https://fueleconomy.gov/feg/epadata/vehicles.csv"'
            f" -o {csv_path}"
        )
        sys.exit(1)

    start_time = time.time()
    process_csv(csv_path, args.min_year)
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f} seconds.")


if __name__ == "__main__":
    main()
