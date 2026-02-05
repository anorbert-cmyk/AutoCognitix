#!/usr/bin/env python3
"""
Engine-Codes.com DTC Database Scraper.

This script scrapes DTC codes from engine-codes.com which contains a comprehensive
database of OBD-II trouble codes with detailed descriptions, causes, and symptoms.

URL Patterns:
    - Code list by prefix: https://www.engine-codes.com/pxxxx.html (P0xxx codes)
    - Individual code: https://www.engine-codes.com/p0xxx.html
    - Code index: https://www.engine-codes.com/obd-codes-a-z.html

The site organizes codes by:
    - P codes (Powertrain) - most comprehensive
    - B codes (Body)
    - C codes (Chassis)
    - U codes (Network)

Usage:
    python scripts/scrape_engine_codes.py --scrape        # Scrape all codes
    python scripts/scrape_engine_codes.py --postgres      # Import to PostgreSQL
    python scripts/scrape_engine_codes.py --all           # Scrape and import
    python scripts/scrape_engine_codes.py --use-cache     # Use cached data
"""

import argparse
import asyncio
import json
import logging
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import httpx
from bs4 import BeautifulSoup
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import shared utilities
from scripts.utils import (
    DatabaseImporter,
    get_category_from_code,
    get_severity_from_code,
    get_system_from_code,
    normalize_dtc_codes,
    sanitize_text,
    setup_logging,
    validate_dtc_code,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Data paths
DATA_DIR = PROJECT_ROOT / "data" / "dtc_codes"
CACHE_FILE = DATA_DIR / "engine_codes.json"

# Engine-codes.com URL configuration
BASE_URL = "https://www.engine-codes.com"

# DTC code prefixes and ranges to scrape
# Engine-codes.com organizes codes in index pages by prefix
CODE_INDEX_PAGES = [
    # Powertrain generic (P0xxx)
    "/p0xxx_codes.html",
    "/p00xx_codes.html",
    "/p01xx_codes.html",
    "/p02xx_codes.html",
    "/p03xx_codes.html",
    "/p04xx_codes.html",
    "/p05xx_codes.html",
    "/p06xx_codes.html",
    "/p07xx_codes.html",
    "/p08xx_codes.html",
    "/p09xx_codes.html",
    # Powertrain manufacturer-specific
    "/p1xxx_codes.html",
    "/p2xxx_codes.html",
    "/p3xxx_codes.html",
    # Body codes
    "/b0xxx_codes.html",
    "/b1xxx_codes.html",
    "/b2xxx_codes.html",
    # Chassis codes
    "/c0xxx_codes.html",
    "/c1xxx_codes.html",
    # Network codes
    "/u0xxx_codes.html",
    "/u1xxx_codes.html",
    "/u2xxx_codes.html",
]

# Alternative: scrape the A-Z index page
INDEX_PAGE = "/obd-codes-a-z.html"

# Rate limiting configuration
RATE_LIMIT_DELAY = 4.0  # seconds between requests (more conservative)
RATE_LIMIT_JITTER = 2.0  # random additional delay (0-2 seconds)
MAX_RETRIES = 3
RETRY_DELAY = 10  # seconds
REQUEST_TIMEOUT = 30  # seconds

# Realistic Chrome User-Agent strings (rotate between these)
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
]

# HTTP headers to appear as a real browser
HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Cache-Control": "max-age=0",
    "Sec-Ch-Ua": '"Not A(Brand";v="99", "Google Chrome";v="121", "Chromium";v="121"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}


class EngineCodesScraper:
    """Scraper for engine-codes.com DTC database."""

    def __init__(self, rate_limit: float = RATE_LIMIT_DELAY):
        """
        Initialize the scraper.

        Args:
            rate_limit: Delay between requests in seconds.
        """
        self.rate_limit = rate_limit
        self.scraped_codes: Dict[str, Dict[str, Any]] = {}
        self.failed_urls: List[str] = []
        self._request_count = 0

    def _get_random_delay(self) -> float:
        """Get a random delay with jitter for more human-like behavior."""
        return self.rate_limit + random.uniform(0, RATE_LIMIT_JITTER)

    def _get_headers(self) -> Dict[str, str]:
        """Get headers with a random User-Agent."""
        headers = HEADERS.copy()
        headers["User-Agent"] = random.choice(USER_AGENTS)
        return headers

    async def _fetch_page(
        self,
        client: httpx.AsyncClient,
        url: str,
        retry_count: int = 0,
    ) -> Optional[str]:
        """
        Fetch a page with retry logic and anti-blocking measures.

        Args:
            client: HTTP client instance.
            url: URL to fetch.
            retry_count: Current retry attempt.

        Returns:
            HTML content or None if fetch failed.
        """
        try:
            self._request_count += 1

            # Use fresh headers with random User-Agent for each request
            headers = self._get_headers()

            # Add referer to make it look more like natural browsing
            if self._request_count > 1:
                headers["Referer"] = BASE_URL + "/"

            response = await client.get(url, headers=headers)

            if response.status_code == 403:
                logger.warning(f"Access forbidden for {url} - may be rate limited (attempt {retry_count + 1})")
                if retry_count < MAX_RETRIES:
                    # Exponential backoff with jitter
                    wait_time = RETRY_DELAY * (2 ** retry_count) + random.uniform(1, 5)
                    logger.info(f"Waiting {wait_time:.1f}s before retry...")
                    await asyncio.sleep(wait_time)
                    return await self._fetch_page(client, url, retry_count + 1)
                return None

            if response.status_code == 404:
                logger.debug(f"Page not found: {url}")
                return None

            if response.status_code != 200:
                logger.warning(f"HTTP {response.status_code} for {url}")
                if retry_count < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY + random.uniform(1, 3))
                    return await self._fetch_page(client, url, retry_count + 1)
                return None

            return response.text

        except httpx.RequestError as e:
            logger.warning(f"Request error for {url}: {e}")
            if retry_count < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY * (retry_count + 1) + random.uniform(1, 3))
                return await self._fetch_page(client, url, retry_count + 1)
        except Exception as e:
            logger.error(f"Unexpected error fetching {url}: {e}")

        return None

    def _parse_code_links_from_index(self, html: str) -> Set[str]:
        """
        Parse DTC code links from an index page.

        Args:
            html: HTML content of the index page.

        Returns:
            Set of code page URLs.
        """
        links = set()
        soup = BeautifulSoup(html, "html.parser")

        # Find all links that match DTC code pattern
        for anchor in soup.find_all("a", href=True):
            href = anchor["href"]

            # Match patterns like /p0100.html, /b0001.html, etc.
            if re.match(r'^/?[pbcu]\d{4}\.html$', href, re.IGNORECASE):
                # Normalize URL
                if not href.startswith("/"):
                    href = "/" + href
                links.add(href)

            # Also check for full URLs
            if "engine-codes.com" in href:
                match = re.search(r'/([pbcu]\d{4})\.html', href, re.IGNORECASE)
                if match:
                    links.add(f"/{match.group(1).lower()}.html")

        logger.debug(f"Found {len(links)} code links in index page")
        return links

    def _parse_individual_code_page(
        self,
        html: str,
        code: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Parse an individual DTC code page for detailed information.

        Engine-codes.com typically has:
        - Code and title at top
        - Description section
        - Possible causes section (often a list)
        - Symptoms section (often a list)
        - Diagnostic steps/tech notes

        Args:
            html: HTML content of the code page.
            code: The DTC code being parsed.

        Returns:
            Dictionary with code data or None if parsing failed.
        """
        soup = BeautifulSoup(html, "html.parser")

        # Initialize data structure
        data = {
            "code": code.upper(),
            "description_en": "",
            "symptoms": [],
            "possible_causes": [],
            "diagnostic_steps": [],
            "source": "engine-codes",
        }

        # Try to find the main content area
        # Common patterns: article, main, div with specific classes
        content_area = (
            soup.find("article") or
            soup.find("main") or
            soup.find("div", class_=re.compile(r'content|article|main', re.I)) or
            soup.find("div", id=re.compile(r'content|article|main', re.I)) or
            soup.body
        )

        if not content_area:
            logger.warning(f"Could not find content area for {code}")
            return None

        # Extract title/description
        # Usually in h1 or h2 tag
        title = content_area.find(["h1", "h2"])
        if title:
            title_text = title.get_text(strip=True)
            # Remove the code from title if present
            title_text = re.sub(rf'^{code}\s*[-:]\s*', '', title_text, flags=re.IGNORECASE)
            data["description_en"] = sanitize_text(title_text)

        # If no title found, try to find description in first paragraph
        if not data["description_en"]:
            first_p = content_area.find("p")
            if first_p:
                data["description_en"] = sanitize_text(first_p.get_text(strip=True)[:500])

        # Extract possible causes
        # Look for sections with "cause" in heading or list after such heading
        causes_section = self._find_section(content_area, ["cause", "reason", "what causes"])
        if causes_section:
            data["possible_causes"] = self._extract_list_items(causes_section)

        # Extract symptoms
        symptoms_section = self._find_section(content_area, ["symptom", "sign", "indication"])
        if symptoms_section:
            data["symptoms"] = self._extract_list_items(symptoms_section)

        # Extract diagnostic steps
        diag_section = self._find_section(
            content_area,
            ["diagnos", "repair", "fix", "solution", "how to", "tech note", "troubleshoot"]
        )
        if diag_section:
            data["diagnostic_steps"] = self._extract_list_items(diag_section)

        # Validate we got at least a description
        if not data["description_en"]:
            logger.warning(f"No description found for {code}")
            return None

        return data

    def _find_section(
        self,
        content: BeautifulSoup,
        keywords: List[str],
    ) -> Optional[BeautifulSoup]:
        """
        Find a section by looking for headings containing keywords.

        Args:
            content: BeautifulSoup content to search.
            keywords: List of keywords to look for in headings.

        Returns:
            The section element or None.
        """
        # Look for headings that contain any of the keywords
        for heading_tag in ["h2", "h3", "h4", "strong", "b"]:
            for heading in content.find_all(heading_tag):
                heading_text = heading.get_text(strip=True).lower()
                if any(kw.lower() in heading_text for kw in keywords):
                    # Return the parent or sibling content
                    # Try to find the next sibling that contains a list or paragraphs
                    next_elem = heading.find_next_sibling()
                    if next_elem:
                        return next_elem

                    # Or try the parent
                    parent = heading.parent
                    if parent:
                        return parent

        return None

    def _extract_list_items(self, element: BeautifulSoup) -> List[str]:
        """
        Extract list items from a section.

        Args:
            element: BeautifulSoup element to extract from.

        Returns:
            List of sanitized text items.
        """
        items = []

        if not element:
            return items

        # First, try to find ul/ol lists
        lists = element.find_all(["ul", "ol"])
        for lst in lists:
            for li in lst.find_all("li"):
                text = sanitize_text(li.get_text(strip=True))
                if text and len(text) > 3:  # Filter out very short items
                    items.append(text)

        # If no lists found, try splitting paragraphs by line breaks or bullet chars
        if not items:
            text = element.get_text(separator="\n", strip=True)
            lines = text.split("\n")
            for line in lines:
                # Clean up bullet characters
                line = re.sub(r'^[\s\-\*\u2022\u25cf\u25cb\u2023]+', '', line).strip()
                line = sanitize_text(line)
                if line and len(line) > 5:
                    items.append(line)

        # Limit to reasonable number of items
        return items[:20]

    async def scrape_index_pages(
        self,
        client: httpx.AsyncClient,
    ) -> Set[str]:
        """
        Scrape all index pages to collect code links.

        Args:
            client: HTTP client instance.

        Returns:
            Set of all code page URLs found.
        """
        all_links = set()

        # First try the main A-Z index
        logger.info("Fetching main index page...")
        html = await self._fetch_page(client, f"{BASE_URL}{INDEX_PAGE}")
        if html:
            links = self._parse_code_links_from_index(html)
            all_links.update(links)
            logger.info(f"Found {len(links)} codes from main index")
            await asyncio.sleep(self._get_random_delay())

        # Then try category-specific index pages
        for index_path in tqdm(CODE_INDEX_PAGES, desc="Scanning index pages"):
            url = f"{BASE_URL}{index_path}"
            html = await self._fetch_page(client, url)

            if html:
                links = self._parse_code_links_from_index(html)
                all_links.update(links)
                logger.debug(f"Found {len(links)} codes from {index_path}")

            await asyncio.sleep(self._get_random_delay())

        logger.info(f"Total unique code pages found: {len(all_links)}")
        return all_links

    async def scrape_code_page(
        self,
        client: httpx.AsyncClient,
        code_path: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Scrape an individual code page.

        Args:
            client: HTTP client instance.
            code_path: URL path to the code page (e.g., "/p0100.html").

        Returns:
            Code data dictionary or None.
        """
        # Extract code from path
        match = re.search(r'/([pbcu]\d{4})\.html', code_path, re.IGNORECASE)
        if not match:
            logger.warning(f"Invalid code path: {code_path}")
            return None

        code = match.group(1).upper()

        # Skip if already scraped
        if code in self.scraped_codes:
            return self.scraped_codes[code]

        url = f"{BASE_URL}{code_path}"
        html = await self._fetch_page(client, url)

        if not html:
            self.failed_urls.append(url)
            return None

        data = self._parse_individual_code_page(html, code)

        if data:
            self.scraped_codes[code] = data

        return data

    async def scrape_all(self) -> List[Dict[str, Any]]:
        """
        Scrape all DTC codes from engine-codes.com.

        Returns:
            List of all scraped code dictionaries.
        """
        logger.info("Starting engine-codes.com scraper...")
        logger.info(f"Using rate limit: {self.rate_limit}s base + 0-{RATE_LIMIT_JITTER}s jitter")

        # Use a session with cookies for more realistic behavior
        async with httpx.AsyncClient(
            timeout=REQUEST_TIMEOUT,
            follow_redirects=True,
            http2=True,  # Use HTTP/2 if available
        ) as client:
            # First, make an initial request to the homepage to get cookies
            logger.info("Warming up session with homepage visit...")
            try:
                homepage_headers = self._get_headers()
                await client.get(BASE_URL + "/", headers=homepage_headers)
                await asyncio.sleep(self._get_random_delay())
            except Exception as e:
                logger.warning(f"Homepage warmup failed: {e}")

            # First, collect all code page URLs
            code_links = await self.scrape_index_pages(client)

            if not code_links:
                # Fallback: generate common code URLs
                logger.info("No index links found, generating code URLs...")
                code_links = self._generate_code_urls()

            # Scrape individual code pages
            logger.info(f"Scraping {len(code_links)} individual code pages...")

            for code_path in tqdm(sorted(code_links), desc="Scraping codes"):
                await self.scrape_code_page(client, code_path)
                await asyncio.sleep(self._get_random_delay())

        # Convert to list and normalize
        codes_list = list(self.scraped_codes.values())
        logger.info(f"Scraped {len(codes_list)} codes, {len(self.failed_urls)} failed")

        return codes_list

    def _generate_code_urls(self) -> Set[str]:
        """
        Generate common DTC code URLs as fallback.

        Returns:
            Set of code page URLs.
        """
        urls = set()

        # Generate P codes (most common)
        for i in range(1000):
            urls.add(f"/p{i:04d}.html")

        # Generate some P1xxx codes
        for i in range(1000, 2000):
            urls.add(f"/p{i:04d}.html")

        # Generate B codes
        for i in range(500):
            urls.add(f"/b{i:04d}.html")

        # Generate C codes
        for i in range(500):
            urls.add(f"/c{i:04d}.html")

        # Generate U codes
        for i in range(500):
            urls.add(f"/u{i:04d}.html")

        return urls


def save_to_cache(codes: List[Dict[str, Any]], file_path: Path) -> None:
    """
    Save codes to a JSON cache file.

    Args:
        codes: List of code dictionaries to save.
        file_path: Path to the cache file.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "metadata": {
            "source": "engine-codes.com",
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "count": len(codes),
        },
        "codes": codes,
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(codes)} codes to {file_path}")


def load_from_cache(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load codes from cache file.

    Args:
        file_path: Path to the cache file.

    Returns:
        List of code dictionaries.
    """
    if not file_path.exists():
        logger.warning(f"Cache file not found: {file_path}")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    codes = data.get("codes", [])
    metadata = data.get("metadata", {})
    logger.info(
        f"Loaded {len(codes)} codes from cache "
        f"(scraped: {metadata.get('scraped_at', 'unknown')})"
    )
    return codes


def merge_with_master(
    new_codes: List[Dict[str, Any]],
    master_file: Path,
) -> tuple[int, int]:
    """
    Merge scraped codes with master codes file.

    Args:
        new_codes: Newly scraped codes.
        master_file: Path to master codes JSON file.

    Returns:
        Tuple of (new_count, updated_count).
    """
    if not master_file.exists():
        logger.warning(f"Master file not found: {master_file}")
        return 0, 0

    with open(master_file, "r", encoding="utf-8") as f:
        master_data = json.load(f)

    master_codes = {c["code"]: c for c in master_data.get("codes", [])}

    new_count = 0
    updated_count = 0

    for code_data in new_codes:
        code = code_data["code"]

        if code in master_codes:
            existing = master_codes[code]

            # Update if new data has more information
            if code_data.get("possible_causes") and not existing.get("possible_causes"):
                existing["possible_causes"] = code_data["possible_causes"]
                updated_count += 1
            if code_data.get("symptoms") and not existing.get("symptoms"):
                existing["symptoms"] = code_data["symptoms"]
                updated_count += 1
            if code_data.get("diagnostic_steps") and not existing.get("diagnostic_steps"):
                existing["diagnostic_steps"] = code_data["diagnostic_steps"]
                updated_count += 1
        else:
            master_codes[code] = code_data
            new_count += 1

    # Save updated master
    master_data["codes"] = sorted(master_codes.values(), key=lambda x: x["code"])
    master_data["metadata"]["count"] = len(master_data["codes"])
    master_data["metadata"]["last_updated"] = datetime.now(timezone.utc).isoformat()

    with open(master_file, "w", encoding="utf-8") as f:
        json.dump(master_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Merged: {new_count} new codes, {updated_count} fields updated")
    return new_count, updated_count


def print_summary(codes: List[Dict[str, Any]]) -> None:
    """Print a summary of scraped codes."""
    print("\n" + "=" * 60)
    print("ENGINE-CODES.COM SCRAPE SUMMARY")
    print("=" * 60)
    print(f"Total codes scraped: {len(codes)}")

    # Count by category
    categories: Dict[str, int] = {}
    with_causes = 0
    with_symptoms = 0
    with_diag = 0

    for code in codes:
        cat = code.get("category", get_category_from_code(code["code"]))
        categories[cat] = categories.get(cat, 0) + 1

        if code.get("possible_causes"):
            with_causes += 1
        if code.get("symptoms"):
            with_symptoms += 1
        if code.get("diagnostic_steps"):
            with_diag += 1

    print("\nBy category:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")

    print("\nData completeness:")
    print(f"  With possible causes: {with_causes} ({100*with_causes//len(codes) if codes else 0}%)")
    print(f"  With symptoms: {with_symptoms} ({100*with_symptoms//len(codes) if codes else 0}%)")
    print(f"  With diagnostic steps: {with_diag} ({100*with_diag//len(codes) if codes else 0}%)")
    print("=" * 60)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Scrape DTC codes from engine-codes.com database"
    )
    parser.add_argument(
        "--scrape",
        action="store_true",
        help="Scrape codes from engine-codes.com website",
    )
    parser.add_argument(
        "--postgres",
        action="store_true",
        help="Import to PostgreSQL",
    )
    parser.add_argument(
        "--neo4j",
        action="store_true",
        help="Import to Neo4j",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Scrape and import to all databases",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use cached data instead of scraping",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge scraped codes with master file (all_codes_merged.json)",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=RATE_LIMIT_DELAY,
        help=f"Delay between requests in seconds (default: {RATE_LIMIT_DELAY})",
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

    # Default to scrape only if nothing specified
    if not (args.scrape or args.postgres or args.neo4j or args.all or args.use_cache):
        args.scrape = True

    try:
        # Get codes
        if args.use_cache:
            codes = load_from_cache(CACHE_FILE)
        elif args.scrape or args.all:
            scraper = EngineCodesScraper(rate_limit=args.rate_limit)
            raw_codes = await scraper.scrape_all()
            codes = normalize_dtc_codes(raw_codes)
            save_to_cache(codes, CACHE_FILE)
        else:
            codes = load_from_cache(CACHE_FILE)

        if not codes:
            logger.warning("No codes available.")
            return

        # Print summary
        print_summary(codes)

        # Merge with master if requested
        if args.merge:
            master_file = DATA_DIR / "all_codes_merged.json"
            merge_with_master(codes, master_file)

        # Import to databases
        if args.postgres or args.all:
            importer = DatabaseImporter()
            inserted, skipped = importer.import_to_postgres(codes)
            logger.info(f"PostgreSQL: {inserted} inserted, {skipped} skipped")

        if args.neo4j or args.all:
            importer = DatabaseImporter()
            created = importer.import_to_neo4j(codes)
            logger.info(f"Neo4j: {created} nodes created")

    except KeyboardInterrupt:
        logger.info("Scraping interrupted by user")
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
