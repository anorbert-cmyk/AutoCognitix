#!/usr/bin/env python3
"""
Back4App Car Make Model Dataset Importer for AutoCognitix

This script imports vehicle make/model data from multiple sources:
1. Primary: Back4App Car Make Model Dataset via Parse API (6000+ models)
2. Fallback: GitHub CarMakesAndModels dataset (1134 models, 2005-2024)

The data is transformed into a unified format and saved to:
    data/vehicles/back4app_vehicles.json

Usage:
    python scripts/import_back4app_vehicles.py
    python scripts/import_back4app_vehicles.py --source back4app
    python scripts/import_back4app_vehicles.py --source github
    python scripts/import_back4app_vehicles.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Output paths
DATA_DIR = PROJECT_ROOT / "data" / "vehicles"
OUTPUT_FILE = DATA_DIR / "back4app_vehicles.json"

# Back4App API Configuration
# Get credentials from: https://www.back4app.com/database/back4app/car-make-model-dataset
# Set BACK4APP_APP_ID and BACK4APP_API_KEY in your .env file
BACK4APP_CONFIG = {
    "app_id": os.getenv("BACK4APP_APP_ID", ""),
    "api_key": os.getenv("BACK4APP_API_KEY", ""),
    "base_url": "https://parseapi.back4app.com/classes",
    "make_class": "Carmodels_Car_Make",
    "model_class": "Carmodels_Car_Model_List",
}

# GitHub Fallback Configuration
GITHUB_CONFIG = {
    "url": "https://raw.githubusercontent.com/demirelarda/CarMakesAndModels/master/carData.json",
    "description": "CarMakesAndModels by demirelarda (2005-2024)",
}


class VehicleImporter:
    """Import vehicle data from Back4App or GitHub fallback."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.session = requests.Session()
        self.stats = {
            "source": None,
            "total_makes": 0,
            "total_models": 0,
            "total_year_entries": 0,
            "import_time": None,
        }

    def fetch_from_back4app(self) -> dict[str, Any] | None:
        """Fetch vehicle data from Back4App Parse API."""
        logger.info("Attempting to fetch from Back4App API...")

        # Validate credentials
        if not BACK4APP_CONFIG["app_id"] or not BACK4APP_CONFIG["api_key"]:
            logger.warning(
                "Back4App credentials not configured. "
                "Set BACK4APP_APP_ID and BACK4APP_API_KEY in .env file. "
                "Get credentials from: https://www.back4app.com/database/back4app/car-make-model-dataset"
            )
            return None

        headers = {
            "X-Parse-Application-Id": BACK4APP_CONFIG["app_id"],
            "X-Parse-REST-API-Key": BACK4APP_CONFIG["api_key"],
        }

        try:
            # First, fetch all makes
            makes_url = f"{BACK4APP_CONFIG['base_url']}/{BACK4APP_CONFIG['make_class']}"
            makes_data = self._fetch_all_pages(makes_url, headers, "makes")

            if not makes_data:
                logger.warning("No makes found from Back4App API")
                return None

            logger.info(f"Fetched {len(makes_data)} makes from Back4App")

            # Build make lookup
            make_lookup = {m["objectId"]: m["Make"] for m in makes_data}

            # Fetch all models
            models_url = f"{BACK4APP_CONFIG['base_url']}/{BACK4APP_CONFIG['model_class']}"
            models_data = self._fetch_all_pages(models_url, headers, "models")

            if not models_data:
                logger.warning("No models found from Back4App API")
                return None

            logger.info(f"Fetched {len(models_data)} models from Back4App")

            # Transform to unified format
            return self._transform_back4app_data(make_lookup, models_data)

        except requests.RequestException as e:
            logger.warning(f"Back4App API request failed: {e}")
            return None
        except (KeyError, json.JSONDecodeError) as e:
            logger.warning(f"Back4App data parsing failed: {e}")
            return None

    def _fetch_all_pages(
        self, url: str, headers: dict, data_type: str
    ) -> list[dict]:
        """Fetch all pages from Back4App API using pagination."""
        all_results = []
        skip = 0
        limit = 1000  # Max allowed by Parse

        with tqdm(desc=f"Fetching {data_type}", unit=" records") as pbar:
            while True:
                params = {"limit": limit, "skip": skip}
                response = self.session.get(
                    url, headers=headers, params=params, timeout=30
                )
                response.raise_for_status()

                data = response.json()
                results = data.get("results", [])

                if not results:
                    break

                all_results.extend(results)
                pbar.update(len(results))
                skip += limit

                # Rate limiting
                time.sleep(0.1)

                if len(results) < limit:
                    break

        return all_results

    def _transform_back4app_data(
        self, make_lookup: dict[str, str], models_data: list[dict]
    ) -> dict[str, Any]:
        """Transform Back4App data to unified format."""
        makes_dict: dict[str, dict] = defaultdict(
            lambda: {"name": "", "models": defaultdict(lambda: {"name": "", "years": []})}
        )

        for model in tqdm(models_data, desc="Processing models"):
            try:
                # Get make info
                make_ptr = model.get("Make", {})
                make_id = make_ptr.get("objectId") if isinstance(make_ptr, dict) else None
                make_name = make_lookup.get(make_id, "Unknown")

                # Get model info
                model_name = model.get("Model", "Unknown")
                year = model.get("Year")

                if make_name and model_name:
                    # Normalize make name as key
                    make_key = make_name.lower().replace(" ", "_").replace("-", "_")
                    makes_dict[make_key]["name"] = make_name

                    # Normalize model name as key
                    model_key = model_name.lower().replace(" ", "_").replace("-", "_")
                    makes_dict[make_key]["models"][model_key]["name"] = model_name

                    if year:
                        year_int = int(year) if isinstance(year, str) else year
                        if year_int not in makes_dict[make_key]["models"][model_key]["years"]:
                            makes_dict[make_key]["models"][model_key]["years"].append(year_int)

            except (ValueError, TypeError, KeyError) as e:
                logger.debug(f"Skipping malformed model entry: {e}")
                continue

        # Convert to final format
        return self._finalize_data(makes_dict, "back4app")

    def fetch_from_github(self) -> dict[str, Any] | None:
        """Fetch vehicle data from GitHub fallback source."""
        logger.info("Fetching from GitHub fallback source...")

        try:
            response = self.session.get(GITHUB_CONFIG["url"], timeout=60)
            response.raise_for_status()
            raw_data = response.json()

            return self._transform_github_data(raw_data)

        except requests.RequestException as e:
            logger.error(f"GitHub fetch failed: {e}")
            return None
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"GitHub data parsing failed: {e}")
            return None

    def _transform_github_data(self, raw_data: dict) -> dict[str, Any]:
        """Transform GitHub CarMakesAndModels data to unified format."""
        makes_dict: dict[str, dict] = defaultdict(
            lambda: {"name": "", "models": defaultdict(lambda: {"name": "", "years": []})}
        )

        years_data = raw_data.get("data", [{}])[0].get("years", [])

        for year_entry in tqdm(years_data, desc="Processing years"):
            year = year_entry.get("year")
            if not year:
                continue

            for make_entry in year_entry.get("makes", []):
                make_name = make_entry.get("makeName", "")
                if not make_name:
                    continue

                # Normalize make key
                make_key = make_name.lower().replace(" ", "_").replace("-", "_")
                makes_dict[make_key]["name"] = make_name

                for model_entry in make_entry.get("models", []):
                    model_name = model_entry.get("modelName", "")
                    if not model_name:
                        continue

                    # Normalize model key
                    model_key = model_name.lower().replace(" ", "_").replace("-", "_")
                    makes_dict[make_key]["models"][model_key]["name"] = model_name

                    if year not in makes_dict[make_key]["models"][model_key]["years"]:
                        makes_dict[make_key]["models"][model_key]["years"].append(year)

        return self._finalize_data(makes_dict, "github")

    def _finalize_data(
        self, makes_dict: dict, source: str
    ) -> dict[str, Any]:
        """Finalize the data structure and calculate statistics."""
        # Convert to list format and sort
        makes_list = []
        total_models = 0
        total_year_entries = 0

        for make_key in sorted(makes_dict.keys()):
            make_data = makes_dict[make_key]
            models_list = []

            for model_key in sorted(make_data["models"].keys()):
                model_data = make_data["models"][model_key]
                years = sorted(model_data["years"])

                models_list.append({
                    "name": model_data["name"],
                    "years": years,
                })
                total_year_entries += len(years)

            makes_list.append({
                "name": make_data["name"],
                "models": models_list,
            })
            total_models += len(models_list)

        # Update stats
        self.stats["source"] = source
        self.stats["total_makes"] = len(makes_list)
        self.stats["total_models"] = total_models
        self.stats["total_year_entries"] = total_year_entries
        self.stats["import_time"] = datetime.utcnow().isoformat()

        # Calculate year range
        all_years = set()
        for make in makes_list:
            for model in make["models"]:
                all_years.update(model["years"])

        year_range = f"{min(all_years)}-{max(all_years)}" if all_years else "N/A"

        return {
            "metadata": {
                "source": source,
                "source_url": BACK4APP_CONFIG["base_url"] if source == "back4app" else GITHUB_CONFIG["url"],
                "description": f"Vehicle make/model database ({source})",
                "total_makes": len(makes_list),
                "total_models": total_models,
                "total_year_entries": total_year_entries,
                "year_range": year_range,
                "imported_at": self.stats["import_time"],
            },
            "makes": makes_list,
        }

    def save_data(self, data: dict[str, Any]) -> bool:
        """Save the transformed data to JSON file."""
        if self.dry_run:
            logger.info("[DRY RUN] Would save data to: %s", OUTPUT_FILE)
            logger.info("[DRY RUN] Data preview:")
            logger.info(json.dumps(data["metadata"], indent=2))
            return True

        try:
            # Ensure directory exists
            DATA_DIR.mkdir(parents=True, exist_ok=True)

            # Write JSON with pretty formatting
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.info(f"Data saved to: {OUTPUT_FILE}")
            return True

        except OSError as e:
            logger.error(f"Failed to save data: {e}")
            return False

    def run(self, source: str = "auto") -> bool:
        """Run the import process."""
        data = None

        if source in ("auto", "back4app"):
            data = self.fetch_from_back4app()

        if data is None and source in ("auto", "github"):
            logger.info("Falling back to GitHub source...")
            data = self.fetch_from_github()

        if data is None:
            logger.error("Failed to fetch data from any source")
            return False

        # Save the data
        if not self.save_data(data):
            return False

        # Print summary
        self._print_summary(data)
        return True

    def _print_summary(self, data: dict[str, Any]) -> None:
        """Print import summary."""
        meta = data["metadata"]
        print("\n" + "=" * 60)
        print("IMPORT SUMMARY")
        print("=" * 60)
        print(f"Source:           {meta['source']}")
        print(f"Total Makes:      {meta['total_makes']}")
        print(f"Total Models:     {meta['total_models']}")
        print(f"Year Entries:     {meta['total_year_entries']}")
        print(f"Year Range:       {meta['year_range']}")
        print(f"Output File:      {OUTPUT_FILE}")
        print("=" * 60)

        # Sample makes
        print("\nSample Makes (first 10):")
        for make in data["makes"][:10]:
            model_count = len(make["models"])
            print(f"  - {make['name']}: {model_count} models")

        print("\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Import Back4App Car Make Model Dataset"
    )
    parser.add_argument(
        "--source",
        choices=["auto", "back4app", "github"],
        default="auto",
        help="Data source to use (default: auto - tries back4app first, then github)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without saving",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    importer = VehicleImporter(dry_run=args.dry_run)
    success = importer.run(source=args.source)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
