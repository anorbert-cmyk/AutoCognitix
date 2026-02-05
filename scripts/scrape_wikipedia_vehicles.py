#!/usr/bin/env python3
"""
Wikipedia Vehicle Data Scraper for AutoCognitix.

This script extracts vehicle technical specifications from Wikipedia
using the MediaWiki API for robots.txt compliance.

Features:
- Uses official Wikipedia API (not scraping)
- Rate limiting (1 request per second by default)
- Caching of results to avoid redundant requests
- Extracts: model names, production years, engines, platforms, body styles

Usage:
    python scripts/scrape_wikipedia_vehicles.py [--verbose] [--no-cache]
"""

import argparse
import hashlib
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, asdict
import urllib.parse

import requests

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "vehicles"
CACHE_DIR = DATA_DIR / "wikipedia_cache"
OUTPUT_FILE = DATA_DIR / "wikipedia_vehicles.json"

# Wikipedia API configuration
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "AutoCognitix/1.0 (https://github.com/autocognitix; contact@autocognitix.com) Python/3.11"

# Rate limiting
DEFAULT_RATE_LIMIT = 1.0  # seconds between requests
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class VehicleModel:
    """Represents a vehicle model with its specifications."""

    make: str
    model: str
    production_years: Optional[str] = None
    years_start: Optional[int] = None
    years_end: Optional[int] = None
    engines: List[str] = None
    platforms: List[str] = None
    body_styles: List[str] = None
    chassis_codes: List[str] = None
    assembly_locations: List[str] = None
    wikipedia_url: str = None
    raw_data: Dict[str, Any] = None

    def __post_init__(self):
        if self.engines is None:
            self.engines = []
        if self.platforms is None:
            self.platforms = []
        if self.body_styles is None:
            self.body_styles = []
        if self.chassis_codes is None:
            self.chassis_codes = []
        if self.assembly_locations is None:
            self.assembly_locations = []
        if self.raw_data is None:
            self.raw_data = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding raw_data for cleaner output."""
        result = asdict(self)
        # Remove raw_data for output
        del result['raw_data']
        return result


class WikipediaRateLimiter:
    """Rate limiter for Wikipedia API requests."""

    def __init__(self, min_interval: float = DEFAULT_RATE_LIMIT):
        self.min_interval = min_interval
        self.last_request_time = 0

    def wait(self):
        """Wait if necessary to respect rate limit."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self.last_request_time = time.time()


class WikipediaCache:
    """Simple file-based cache for Wikipedia API responses."""

    def __init__(self, cache_dir: Path, enabled: bool = True):
        self.cache_dir = cache_dir
        self.enabled = enabled
        if enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for a given key."""
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{hash_key}.json"

    def get(self, key: str) -> Optional[Dict]:
        """Get cached value if available."""
        if not self.enabled:
            return None

        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Check if cache is fresh (24 hours)
                    cached_time = data.get('_cached_at', 0)
                    if time.time() - cached_time < 86400:
                        logger.debug(f"Cache hit for: {key[:50]}...")
                        return data.get('data')
            except (json.JSONDecodeError, IOError):
                pass
        return None

    def set(self, key: str, value: Dict):
        """Store value in cache."""
        if not self.enabled:
            return

        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump({
                    '_cached_at': time.time(),
                    'data': value
                }, f)
        except IOError as e:
            logger.warning(f"Failed to write cache: {e}")


class WikipediaVehicleScraper:
    """Scraper for vehicle data from Wikipedia using the MediaWiki API."""

    # Major vehicle manufacturers to scrape
    MANUFACTURERS = [
        ("Volkswagen", "List of Volkswagen vehicles"),
        ("BMW", "List of BMW vehicles"),
        ("Toyota", "List of Toyota vehicles"),
        ("Mercedes-Benz", "List of Mercedes-Benz vehicles"),
        ("Audi", "List of Audi vehicles"),
        ("Ford", "List of Ford vehicles"),
        ("Honda", "List of Honda vehicles"),
        ("Nissan", "List of Nissan vehicles"),
        ("Hyundai", "List of Hyundai vehicles"),
        ("Kia", "List of Kia vehicles"),
        ("Chevrolet", "List of Chevrolet vehicles"),
        ("Mazda", "List of Mazda vehicles"),
        ("Subaru", "List of Subaru vehicles"),
        ("Volvo", "List of Volvo vehicles"),
        ("Porsche", "List of Porsche vehicles"),
        ("Fiat", "List of Fiat vehicles"),
        ("Peugeot", "List of Peugeot vehicles"),
        ("Renault", "List of Renault vehicles"),
        ("Opel", "List of Opel vehicles"),
        ("Skoda", "List of Skoda vehicles"),
        ("SEAT", "List of SEAT vehicles"),
        ("Suzuki", "List of Suzuki vehicles"),
        ("Mitsubishi", "List of Mitsubishi vehicles"),
        ("Lexus", "List of Lexus vehicles"),
        ("Infiniti", "List of Infiniti vehicles"),
        ("Acura", "List of Acura vehicles"),
        ("Jeep", "List of Jeep vehicles"),
        ("Dodge", "List of Dodge vehicles"),
        ("Chrysler", "List of Chrysler vehicles"),
        ("Tesla", "List of Tesla vehicles"),
    ]

    def __init__(self, rate_limit: float = DEFAULT_RATE_LIMIT, use_cache: bool = True):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': USER_AGENT,
            'Accept': 'application/json',
        })
        self.rate_limiter = WikipediaRateLimiter(rate_limit)
        self.cache = WikipediaCache(CACHE_DIR, enabled=use_cache)
        self.vehicles: List[VehicleModel] = []

    def _api_request(self, params: Dict[str, str]) -> Optional[Dict]:
        """Make a rate-limited request to the Wikipedia API."""
        params.setdefault('format', 'json')
        params.setdefault('formatversion', '2')

        cache_key = json.dumps(params, sort_keys=True)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        self.rate_limiter.wait()

        for attempt in range(MAX_RETRIES):
            try:
                response = self.session.get(WIKIPEDIA_API_URL, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                self.cache.set(cache_key, data)
                return data
            except requests.RequestException as e:
                logger.warning(f"API request failed (attempt {attempt + 1}): {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    logger.error(f"Failed after {MAX_RETRIES} attempts")
                    return None
        return None

    def get_page_content(self, title: str) -> Optional[str]:
        """Get the wikitext content of a page."""
        params = {
            'action': 'query',
            'titles': title,
            'prop': 'revisions',
            'rvprop': 'content',
            'rvslots': 'main',
        }

        data = self._api_request(params)
        if not data:
            return None

        pages = data.get('query', {}).get('pages', [])
        if not pages:
            return None

        page = pages[0]
        if 'missing' in page:
            logger.warning(f"Page not found: {title}")
            return None

        revisions = page.get('revisions', [])
        if not revisions:
            return None

        return revisions[0].get('slots', {}).get('main', {}).get('content', '')

    def get_page_links(self, title: str) -> List[str]:
        """Get all internal links from a page."""
        links = []
        params = {
            'action': 'query',
            'titles': title,
            'prop': 'links',
            'pllimit': 'max',
        }

        while True:
            data = self._api_request(params)
            if not data:
                break

            pages = data.get('query', {}).get('pages', [])
            for page in pages:
                for link in page.get('links', []):
                    links.append(link.get('title', ''))

            # Handle pagination
            if 'continue' in data:
                params['plcontinue'] = data['continue']['plcontinue']
            else:
                break

        return links

    def parse_vehicle_infobox(self, content: str, make: str, model_name: str) -> VehicleModel:
        """Parse vehicle specifications from a Wikipedia infobox."""
        vehicle = VehicleModel(make=make, model=model_name)

        if not content:
            return vehicle

        # Extract infobox
        infobox_match = re.search(r'\{\{Infobox automobile(.*?)\n\}\}', content, re.DOTALL | re.IGNORECASE)
        if not infobox_match:
            infobox_match = re.search(r'\{\{Infobox car(.*?)\n\}\}', content, re.DOTALL | re.IGNORECASE)

        if infobox_match:
            infobox_text = infobox_match.group(1)
            vehicle.raw_data['infobox'] = infobox_text[:2000]  # Limit size

            # Extract production years
            years_match = re.search(r'\|\s*production\s*=\s*([^\n|]+)', infobox_text, re.IGNORECASE)
            if years_match:
                years_text = self._clean_wiki_text(years_match.group(1))
                vehicle.production_years = years_text
                vehicle.years_start, vehicle.years_end = self._parse_year_range(years_text)

            # Extract engines
            engine_match = re.search(r'\|\s*engine\s*=\s*([^\n]+(?:\n\s*\*[^\n]+)*)', infobox_text, re.IGNORECASE)
            if engine_match:
                engines_text = self._clean_wiki_text(engine_match.group(1))
                vehicle.engines = self._parse_list_field(engines_text)

            # Extract platform/chassis
            platform_match = re.search(r'\|\s*platform\s*=\s*([^\n|]+)', infobox_text, re.IGNORECASE)
            if platform_match:
                platform_text = self._clean_wiki_text(platform_match.group(1))
                vehicle.platforms = self._parse_list_field(platform_text)

            # Extract chassis codes
            chassis_match = re.search(r'\|\s*(?:chassis|internal_code|model_code)\s*=\s*([^\n|]+)', infobox_text, re.IGNORECASE)
            if chassis_match:
                chassis_text = self._clean_wiki_text(chassis_match.group(1))
                vehicle.chassis_codes = self._parse_list_field(chassis_text)

            # Extract body styles
            body_match = re.search(r'\|\s*(?:body_style|body)\s*=\s*([^\n|]+)', infobox_text, re.IGNORECASE)
            if body_match:
                body_text = self._clean_wiki_text(body_match.group(1))
                vehicle.body_styles = self._parse_list_field(body_text)

            # Extract assembly locations
            assembly_match = re.search(r'\|\s*assembly\s*=\s*([^\n]+(?:\n\s*\*[^\n]+)*)', infobox_text, re.IGNORECASE)
            if assembly_match:
                assembly_text = self._clean_wiki_text(assembly_match.group(1))
                vehicle.assembly_locations = self._parse_list_field(assembly_text)

        return vehicle

    def _clean_wiki_text(self, text: str) -> str:
        """Remove wiki markup from text."""
        if not text:
            return ""

        # Remove nested templates first (handle {{ubl|...|...}})
        max_iterations = 10
        for _ in range(max_iterations):
            new_text = re.sub(r'\{\{[^{}]*\}\}', '', text)
            if new_text == text:
                break
            text = new_text

        # Remove any remaining template markers
        text = re.sub(r'\{\{[^}]*\}\}?', '', text)
        text = re.sub(r'\}\}', '', text)

        # Remove wiki links [[link|text]] -> text or [[link]] -> link
        text = re.sub(r'\[\[([^|\]]+)\|([^\]]+)\]\]', r'\2', text)
        text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)

        # Remove references <ref>...</ref>
        text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
        text = re.sub(r'<ref[^>]*/>', '', text)

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Remove '''bold''' and ''italic''
        text = re.sub(r"'''?", '', text)

        # Remove {{Cite ...}} patterns that may remain
        text = re.sub(r'Cite\s+\w+', '', text)

        # Clean up whitespace and special chars
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'^\s*[|*]\s*', '', text)

        return text.strip()

    def _parse_year_range(self, text: str) -> tuple[Optional[int], Optional[int]]:
        """Parse year range from text like '1998-2005' or '2010-present'."""
        if not text:
            return None, None

        # Match patterns like "1998-2005", "1998 - 2005", "1998-present"
        match = re.search(r'(\d{4})\s*[-–]\s*(\d{4}|present|current)?', text, re.IGNORECASE)
        if match:
            start_year = int(match.group(1))
            end_text = match.group(2)
            if end_text and end_text.lower() not in ('present', 'current'):
                try:
                    end_year = int(end_text)
                except ValueError:
                    end_year = None
            else:
                end_year = None  # Still in production
            return start_year, end_year

        # Match single year
        match = re.search(r'(\d{4})', text)
        if match:
            return int(match.group(1)), None

        return None, None

    def _parse_list_field(self, text: str) -> List[str]:
        """Parse a comma/semicolon/newline separated list from text."""
        if not text:
            return []

        # Split by common separators
        items = re.split(r'[,;\n*]', text)

        # Clean and filter
        result = []
        for item in items:
            item = item.strip()
            if item and len(item) > 1 and not item.startswith('|'):
                # Skip obvious junk
                if not re.match(r'^[\d\s.]+$', item):
                    result.append(item)

        return list(dict.fromkeys(result))  # Remove duplicates while preserving order

    def scrape_manufacturer_list(self, make: str, list_page: str) -> List[VehicleModel]:
        """Scrape all vehicles from a manufacturer's vehicle list page."""
        logger.info(f"Scraping {make} vehicles from: {list_page}")

        content = self.get_page_content(list_page)
        if not content:
            logger.warning(f"Could not get content for {list_page}")
            return []

        vehicles = []

        # Extract vehicle model names from the list page
        # Look for table entries and list items
        model_names = self._extract_model_names(content, make)
        logger.info(f"Found {len(model_names)} potential models for {make}")

        for model_name in model_names:
            # Construct the expected Wikipedia page title
            page_title = f"{make} {model_name}"

            # Get vehicle page content
            vehicle_content = self.get_page_content(page_title)
            if vehicle_content:
                vehicle = self.parse_vehicle_infobox(vehicle_content, make, model_name)
                vehicle.wikipedia_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(page_title.replace(' ', '_'))}"
                vehicles.append(vehicle)
                logger.debug(f"Parsed: {make} {model_name}")
            else:
                # Try alternative page titles
                for alt_title in [model_name, f"{model_name} ({make})", f"{make}_{model_name}"]:
                    vehicle_content = self.get_page_content(alt_title)
                    if vehicle_content:
                        vehicle = self.parse_vehicle_infobox(vehicle_content, make, model_name)
                        vehicle.wikipedia_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(alt_title.replace(' ', '_'))}"
                        vehicles.append(vehicle)
                        logger.debug(f"Parsed (alt): {make} {model_name}")
                        break
                else:
                    # Create basic entry without detailed info
                    vehicle = VehicleModel(make=make, model=model_name)
                    vehicles.append(vehicle)
                    logger.debug(f"Basic entry: {make} {model_name}")

        return vehicles

    def _extract_model_names(self, content: str, make: str) -> List[str]:
        """Extract vehicle model names from a list page."""
        models = set()

        # Pattern 1: Wiki links that likely refer to vehicle models
        # [[Make Model]] or [[Make Model|Display Text]]
        pattern1 = re.findall(
            rf'\[\[{re.escape(make)}\s+([^|\]]+?)(?:\s*\([^)]+\))?\s*(?:\|[^\]]+)?\]\]',
            content,
            re.IGNORECASE
        )
        models.update(pattern1)

        # Pattern 2: Links without make prefix that might be models
        # Common pattern in vehicle list tables
        all_links = re.findall(r'\[\[([^|\]]+)(?:\|[^\]]+)?\]\]', content)

        for link in all_links:
            link = link.strip()
            # Skip if it's a category, file, or generic page
            if any(skip in link.lower() for skip in ['category:', 'file:', 'image:', 'wikipedia:', 'template:']):
                continue
            # Skip if too long (likely a sentence, not a model name)
            if len(link) > 50:
                continue
            # Check if it looks like a vehicle model
            if make.lower() in link.lower():
                # Extract model name
                model = re.sub(rf'^{re.escape(make)}\s*', '', link, flags=re.IGNORECASE).strip()
                if model and len(model) > 1:
                    models.add(model)

        # Pattern 3: Table cell patterns (common in vehicle lists)
        table_cells = re.findall(r'\|\s*\[\[([^\]|]+)', content)
        for cell in table_cells:
            cell = cell.strip()
            if len(cell) < 50 and not any(skip in cell.lower() for skip in ['category:', 'file:']):
                if make.lower() in cell.lower():
                    model = re.sub(rf'^{re.escape(make)}\s*', '', cell, flags=re.IGNORECASE).strip()
                    if model:
                        models.add(model)
                elif not ' ' in cell or len(cell) < 20:
                    # Short names might be model names
                    models.add(cell)

        # Clean up model names
        cleaned = []
        for model in models:
            # Remove parenthetical notes
            model = re.sub(r'\s*\([^)]*\)\s*$', '', model)
            # Remove generation indicators that aren't part of the name
            model = model.strip()
            if model and len(model) > 1 and model.lower() != make.lower():
                cleaned.append(model)

        # Sort and deduplicate
        cleaned = sorted(set(cleaned))

        return cleaned[:100]  # Limit to 100 models per make to avoid timeouts

    def scrape_all_manufacturers(self) -> List[VehicleModel]:
        """Scrape vehicles from all configured manufacturers."""
        all_vehicles = []

        for make, list_page in self.MANUFACTURERS:
            try:
                vehicles = self.scrape_manufacturer_list(make, list_page)
                all_vehicles.extend(vehicles)
                logger.info(f"Scraped {len(vehicles)} vehicles for {make}")
            except Exception as e:
                logger.error(f"Error scraping {make}: {e}")
                continue

        self.vehicles = all_vehicles
        return all_vehicles

    def save_results(self, output_path: Path = OUTPUT_FILE):
        """Save scraped vehicles to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            "metadata": {
                "source": "Wikipedia (MediaWiki API)",
                "scraped_at": datetime.now(timezone.utc).isoformat(),
                "total_vehicles": len(self.vehicles),
                "manufacturers": len(self.MANUFACTURERS),
            },
            "vehicles": [v.to_dict() for v in self.vehicles]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(self.vehicles)} vehicles to {output_path}")
        return output_path

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about scraped vehicles."""
        if not self.vehicles:
            return {}

        makes = {}
        with_years = 0
        with_engines = 0
        with_platforms = 0
        with_body_styles = 0

        for v in self.vehicles:
            makes[v.make] = makes.get(v.make, 0) + 1
            if v.years_start:
                with_years += 1
            if v.engines:
                with_engines += 1
            if v.platforms:
                with_platforms += 1
            if v.body_styles:
                with_body_styles += 1

        return {
            "total_vehicles": len(self.vehicles),
            "vehicles_by_make": makes,
            "with_production_years": with_years,
            "with_engine_info": with_engines,
            "with_platform_info": with_platforms,
            "with_body_styles": with_body_styles,
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Scrape vehicle data from Wikipedia")
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching')
    parser.add_argument('--rate-limit', type=float, default=DEFAULT_RATE_LIMIT,
                        help=f'Seconds between requests (default: {DEFAULT_RATE_LIMIT})')
    parser.add_argument('--makes', nargs='+', help='Only scrape specific makes')
    parser.add_argument('--output', type=Path, default=OUTPUT_FILE, help='Output file path')
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("Starting Wikipedia vehicle scraper")

    # Initialize scraper
    scraper = WikipediaVehicleScraper(
        rate_limit=args.rate_limit,
        use_cache=not args.no_cache
    )

    # Filter makes if specified
    if args.makes:
        scraper.MANUFACTURERS = [
            (make, page) for make, page in scraper.MANUFACTURERS
            if make.lower() in [m.lower() for m in args.makes]
        ]
        if not scraper.MANUFACTURERS:
            logger.error(f"No matching manufacturers found for: {args.makes}")
            return 1

    # Scrape all manufacturers
    vehicles = scraper.scrape_all_manufacturers()

    # Save results
    scraper.save_results(args.output)

    # Print statistics
    stats = scraper.get_statistics()
    logger.info("=" * 50)
    logger.info("SCRAPING COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Total vehicles: {stats.get('total_vehicles', 0)}")
    logger.info(f"With production years: {stats.get('with_production_years', 0)}")
    logger.info(f"With engine info: {stats.get('with_engine_info', 0)}")
    logger.info(f"With platform info: {stats.get('with_platform_info', 0)}")
    logger.info(f"With body styles: {stats.get('with_body_styles', 0)}")
    logger.info("-" * 50)
    logger.info("Vehicles by make:")
    for make, count in sorted(stats.get('vehicles_by_make', {}).items()):
        logger.info(f"  {make}: {count}")

    return 0


if __name__ == "__main__":
    exit(main())
