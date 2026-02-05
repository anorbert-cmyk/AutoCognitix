#!/usr/bin/env python3
"""
OBD-Codes.com DTC Database Scraper.

Scrapes DTC codes from obd-codes.com which has comprehensive OBD-II trouble code listings.

URL Patterns:
    - Generic codes: https://www.obd-codes.com/p0001
    - List pages: https://www.obd-codes.com/trouble_codes/

Usage:
    python scripts/scrape_obd_codes.py --scrape
    python scripts/scrape_obd_codes.py --merge
"""

import argparse
import asyncio
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils import (
    sanitize_text,
    validate_dtc_code,
    get_category_from_code,
    get_severity_from_code,
    get_system_from_code,
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
CACHE_FILE = DATA_DIR / "obd_codes_com.json"
MASTER_FILE = DATA_DIR / "all_codes_merged.json"

# OBD-Codes.com configuration
BASE_URL = "https://www.obd-codes.com"

# Rate limiting
RATE_LIMIT_DELAY = 2.0  # seconds between requests
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# User agent for polite scraping
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

# DTC code prefixes and ranges to scrape
DTC_PREFIXES = ["P", "B", "C", "U"]


async def check_robots_txt(client: httpx.AsyncClient) -> RobotFileParser:
    """
    Fetch and parse robots.txt for obd-codes.com.

    Args:
        client: HTTP client instance.

    Returns:
        Configured RobotFileParser instance.
    """
    robots_url = f"{BASE_URL}/robots.txt"
    rp = RobotFileParser()
    rp.set_url(robots_url)

    try:
        response = await client.get(robots_url)
        if response.status_code == 200:
            # Parse the robots.txt content
            rp.parse(response.text.splitlines())
            logger.info("Successfully loaded robots.txt")
        else:
            logger.warning(f"Could not fetch robots.txt (HTTP {response.status_code})")
            # Default to permissive if robots.txt not found
            rp.parse([])
    except Exception as e:
        logger.warning(f"Error fetching robots.txt: {e}")
        rp.parse([])

    return rp


def is_url_allowed(rp: RobotFileParser, url: str) -> bool:
    """
    Check if a URL is allowed by robots.txt.

    Args:
        rp: RobotFileParser instance.
        url: URL to check.

    Returns:
        True if URL is allowed, False otherwise.
    """
    try:
        # Check against our user agent
        return rp.can_fetch("*", url)
    except Exception:
        return True  # Default to allowed on error


def validate_url(url: str) -> bool:
    """
    Validate that a URL is safe to request.

    Args:
        url: URL to validate.

    Returns:
        True if URL is valid and safe, False otherwise.
    """
    try:
        parsed = urlparse(url)
        # Only allow HTTPS to obd-codes.com
        if parsed.scheme not in ("http", "https"):
            return False
        # Secure domain validation - must match exactly or be a proper subdomain
        # This prevents "attacker-obd-codes.com" from passing validation
        netloc = parsed.netloc.lower()
        allowed_domain = "obd-codes.com"
        if netloc != allowed_domain and not netloc.endswith("." + allowed_domain):
            return False
        # Block any path traversal attempts
        if ".." in parsed.path or "//" in parsed.path[1:]:
            return False
        return True
    except Exception:
        return False


# Generate all possible DTC codes for scraping
def generate_dtc_codes() -> List[str]:
    """Generate list of all possible generic DTC codes to scrape."""
    codes = []

    # P codes: P0000-P0999 (generic), P1000-P3499 (manufacturer specific)
    for i in range(0, 1000):
        codes.append(f"P{i:04d}")
    for i in range(1000, 3500):
        codes.append(f"P{i:04d}")

    # B codes: B0000-B0999 (generic), B1000-B3499 (manufacturer specific)
    for i in range(0, 1000):
        codes.append(f"B{i:04d}")
    for i in range(1000, 3500):
        codes.append(f"B{i:04d}")

    # C codes: C0000-C0999 (generic), C1000-C3499 (manufacturer specific)
    for i in range(0, 1000):
        codes.append(f"C{i:04d}")
    for i in range(1000, 3500):
        codes.append(f"C{i:04d}")

    # U codes: U0000-U0499 (generic), U1000-U3199 (manufacturer specific)
    for i in range(0, 500):
        codes.append(f"U{i:04d}")
    for i in range(1000, 3200):
        codes.append(f"U{i:04d}")

    return codes


async def scrape_code_page(
    client: httpx.AsyncClient,
    code: str,
    rp: RobotFileParser,
    retry_count: int = 0,
) -> Optional[Dict[str, Any]]:
    """
    Scrape a single DTC code page.

    Args:
        client: HTTP client instance.
        code: DTC code to scrape (e.g., "P0001").
        rp: RobotFileParser for checking allowed paths.
        retry_count: Current retry attempt.

    Returns:
        DTC code dictionary or None if not found.
    """
    url = f"{BASE_URL}/{code.lower()}"

    # Security: validate URL before requesting
    if not validate_url(url):
        logger.warning(f"Invalid URL blocked: {url}")
        return None

    # Respect robots.txt
    if not is_url_allowed(rp, url):
        logger.debug(f"URL disallowed by robots.txt: {url}")
        return None

    try:
        response = await client.get(url)

        if response.status_code == 404:
            return None

        if response.status_code == 403:
            logger.warning(f"Access forbidden for {code} - rate limited")
            if retry_count < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY * (retry_count + 1))
                return await scrape_code_page(client, code, rp, retry_count + 1)
            return None

        if response.status_code != 200:
            logger.debug(f"HTTP {response.status_code} for {code}")
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        # Find the main content
        # OBD-Codes.com typically has the description in an article or main content div
        description = None

        # Try multiple selectors for the description
        selectors = [
            "article p:first-of-type",
            ".entry-content p:first-of-type",
            "#content p:first-of-type",
            "main p:first-of-type",
            ".post-content p:first-of-type",
        ]

        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                text = element.get_text(strip=True)
                if len(text) > 20 and code.upper() in text.upper():
                    description = text
                    break

        # Fallback: look for any paragraph containing the code
        if not description:
            for p in soup.find_all("p"):
                text = p.get_text(strip=True)
                if code.upper() in text.upper() and len(text) > 20:
                    description = text
                    break

        # Try to find it in the title
        if not description:
            title = soup.find("title")
            if title:
                title_text = title.get_text(strip=True)
                if code.upper() in title_text.upper():
                    # Extract description from title (often "P0001 - Description")
                    match = re.search(rf'{code}\s*[-:]\s*(.+)', title_text, re.IGNORECASE)
                    if match:
                        description = match.group(1).strip()

        # Look for h1 with description
        if not description:
            h1 = soup.find("h1")
            if h1:
                h1_text = h1.get_text(strip=True)
                match = re.search(rf'{code}\s*[-:]\s*(.+)', h1_text, re.IGNORECASE)
                if match:
                    description = match.group(1).strip()

        if not description:
            return None

        # Clean up description
        description = sanitize_text(description, max_length=500)

        # Remove the code from the beginning if present
        description = re.sub(rf'^{code}\s*[-:]\s*', '', description, flags=re.IGNORECASE)

        if len(description) < 10:
            return None

        # Try to extract symptoms and causes
        symptoms = []
        causes = []

        # Look for symptom sections
        symptom_section = soup.find(string=re.compile(r'symptom', re.IGNORECASE))
        if symptom_section:
            parent = symptom_section.find_parent()
            if parent:
                ul = parent.find_next_sibling("ul")
                if ul:
                    for li in ul.find_all("li"):
                        symptoms.append(sanitize_text(li.get_text(strip=True), max_length=200))

        # Look for cause sections
        cause_section = soup.find(string=re.compile(r'cause|reason', re.IGNORECASE))
        if cause_section:
            parent = cause_section.find_parent()
            if parent:
                ul = parent.find_next_sibling("ul")
                if ul:
                    for li in ul.find_all("li"):
                        causes.append(sanitize_text(li.get_text(strip=True), max_length=200))

        return {
            "code": code.upper(),
            "description_en": description,
            "symptoms": symptoms[:5],  # Limit to 5 items
            "possible_causes": causes[:5],
            "source": "obd-codes.com",
        }

    except httpx.RequestError as e:
        logger.debug(f"Request error for {code}: {e}")
        if retry_count < MAX_RETRIES:
            await asyncio.sleep(RETRY_DELAY * (retry_count + 1))
            return await scrape_code_page(client, code, rp, retry_count + 1)
    except Exception as e:
        logger.error(f"Error scraping {code}: {e}")

    return None


async def scrape_list_pages(
    client: httpx.AsyncClient,
    rp: RobotFileParser,
) -> List[Dict[str, Any]]:
    """
    Scrape the trouble codes list pages to get codes with descriptions.

    Args:
        client: HTTP client instance.
        rp: RobotFileParser for checking allowed paths.

    Returns:
        List of DTC code dictionaries.
    """
    codes = []

    # Known list page patterns on obd-codes.com
    list_urls = [
        f"{BASE_URL}/trouble_codes/",
        f"{BASE_URL}/p0001-p0099",
        f"{BASE_URL}/p0100-p0199",
        f"{BASE_URL}/p0200-p0299",
        f"{BASE_URL}/p0300-p0399",
        f"{BASE_URL}/p0400-p0499",
        f"{BASE_URL}/p0500-p0599",
        f"{BASE_URL}/p0600-p0899",
    ]

    for url in list_urls:
        # Security: validate URL before requesting
        if not validate_url(url):
            logger.warning(f"Invalid URL blocked: {url}")
            continue

        # Respect robots.txt
        if not is_url_allowed(rp, url):
            logger.debug(f"URL disallowed by robots.txt: {url}")
            continue

        try:
            response = await client.get(url)
            if response.status_code != 200:
                continue

            soup = BeautifulSoup(response.text, "html.parser")

            # Find all links that look like DTC codes
            for a in soup.find_all("a"):
                href = a.get("href", "")
                text = a.get_text(strip=True)

                # Check if href points to a code page
                match = re.search(r'/([PCBU][0-9]{4})/?$', href, re.IGNORECASE)
                if match:
                    code = match.group(1).upper()

                    # Try to get description from surrounding text
                    description = ""
                    parent = a.find_parent(["li", "td", "div"])
                    if parent:
                        description = parent.get_text(strip=True)
                        description = re.sub(rf'^{code}\s*[-:]\s*', '', description, flags=re.IGNORECASE)

                    if description and len(description) > 10:
                        codes.append({
                            "code": code,
                            "description_en": sanitize_text(description, max_length=500),
                            "source": "obd-codes.com",
                        })

            await asyncio.sleep(RATE_LIMIT_DELAY)

        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")

    return codes


async def scrape_all_codes(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Scrape all DTC codes from obd-codes.com.

    Args:
        limit: Optional limit on number of codes to scrape.

    Returns:
        List of scraped DTC code dictionaries.
    """
    logger.info("Starting obd-codes.com scrape...")

    all_codes = []
    codes_to_scrape = generate_dtc_codes()

    if limit:
        codes_to_scrape = codes_to_scrape[:limit]

    logger.info(f"Scraping {len(codes_to_scrape)} potential codes...")

    async with httpx.AsyncClient(timeout=30.0, headers=HEADERS, follow_redirects=True) as client:
        # Check robots.txt first - IMPORTANT: Respect site policies
        logger.info("Checking robots.txt...")
        rp = await check_robots_txt(client)

        # First try list pages for quick bulk data
        list_codes = await scrape_list_pages(client, rp)
        logger.info(f"Got {len(list_codes)} codes from list pages")
        all_codes.extend(list_codes)

        # Track which codes we already have
        existing_codes = {c["code"] for c in all_codes}

        # Then scrape individual code pages for ones we don't have
        remaining_codes = [c for c in codes_to_scrape if c not in existing_codes]

        # Sample remaining codes to avoid excessive scraping
        # Focus on generic codes (X0XXX format)
        priority_codes = [c for c in remaining_codes if c[1] == "0"]

        for code in tqdm(priority_codes[:500], desc="Scraping obd-codes.com"):
            if code in existing_codes:
                continue

            code_data = await scrape_code_page(client, code, rp)
            if code_data:
                all_codes.append(code_data)
                existing_codes.add(code)

            await asyncio.sleep(RATE_LIMIT_DELAY)

    logger.info(f"Scraped {len(all_codes)} codes total from obd-codes.com")
    return all_codes


def normalize_codes(codes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize and deduplicate scraped codes."""
    seen = set()
    normalized = []

    for code_data in codes:
        code = code_data.get("code", "").upper().strip()

        if not code or code in seen:
            continue

        if not validate_dtc_code(code):
            continue

        seen.add(code)

        description = sanitize_text(code_data.get("description_en", ""))
        if not description or len(description) < 10:
            continue

        normalized.append({
            "code": code,
            "description_en": description,
            "description_hu": None,
            "category": get_category_from_code(code),
            "severity": get_severity_from_code(code),
            "system": get_system_from_code(code),
            "is_generic": code[1] == "0",
            "symptoms": code_data.get("symptoms", []),
            "possible_causes": code_data.get("possible_causes", []),
            "diagnostic_steps": [],
            "related_codes": [],
            "sources": ["obd-codes.com"],
            "manufacturer": None,
        })

    normalized.sort(key=lambda x: x["code"])
    logger.info(f"Normalized to {len(normalized)} unique codes")
    return normalized


def save_to_cache(codes: List[Dict[str, Any]]) -> None:
    """Save codes to cache file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    data = {
        "metadata": {
            "source": "obd-codes.com",
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "count": len(codes),
        },
        "codes": codes,
    }

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(codes)} codes to {CACHE_FILE}")


def load_from_cache() -> List[Dict[str, Any]]:
    """Load codes from cache file."""
    if not CACHE_FILE.exists():
        return []

    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    codes = data.get("codes", [])
    logger.info(f"Loaded {len(codes)} codes from cache")
    return codes


def merge_with_master(new_codes: List[Dict[str, Any]]) -> tuple[int, int]:
    """
    Merge scraped codes with master file.

    Returns:
        Tuple of (new_count, updated_count).
    """
    if not MASTER_FILE.exists():
        logger.warning(f"Master file not found: {MASTER_FILE}")
        return 0, 0

    with open(MASTER_FILE, "r", encoding="utf-8") as f:
        master_data = json.load(f)

    master_codes = {c["code"]: c for c in master_data.get("codes", [])}

    new_count = 0
    updated_count = 0

    for code_data in new_codes:
        code = code_data["code"]

        if code in master_codes:
            existing = master_codes[code]

            # Add source if not already present
            if "sources" not in existing:
                existing["sources"] = []
            if "obd-codes.com" not in existing["sources"]:
                existing["sources"].append("obd-codes.com")

            # Update if new description is longer
            if len(code_data.get("description_en", "")) > len(existing.get("description_en", "")):
                existing["description_en"] = code_data["description_en"]
                updated_count += 1

            # Merge symptoms and causes
            for field in ["symptoms", "possible_causes"]:
                if code_data.get(field):
                    existing_items = set(existing.get(field, []))
                    for item in code_data[field]:
                        if item not in existing_items:
                            existing.setdefault(field, []).append(item)
        else:
            master_codes[code] = code_data
            new_count += 1

    # Save updated master
    master_data["codes"] = sorted(master_codes.values(), key=lambda x: x["code"])
    master_data["metadata"]["total_codes"] = len(master_data["codes"])
    master_data["metadata"]["generated_at"] = datetime.now(timezone.utc).isoformat()

    with open(MASTER_FILE, "w", encoding="utf-8") as f:
        json.dump(master_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Merged: {new_count} new, {updated_count} updated")
    return new_count, updated_count


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Scrape DTC codes from obd-codes.com")
    parser.add_argument("--scrape", action="store_true", help="Scrape codes from website")
    parser.add_argument("--merge", action="store_true", help="Merge with master file")
    parser.add_argument("--use-cache", action="store_true", help="Use cached data")
    parser.add_argument("--limit", type=int, help="Limit number of codes to scrape")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Default to scrape and merge
    if not (args.scrape or args.merge or args.use_cache):
        args.scrape = True
        args.merge = True

    try:
        if args.use_cache:
            codes = load_from_cache()
        elif args.scrape:
            raw_codes = await scrape_all_codes(limit=args.limit)
            codes = normalize_codes(raw_codes)
            save_to_cache(codes)
        else:
            codes = load_from_cache()

        if not codes:
            logger.warning("No codes available.")
            return

        logger.info(f"Total codes: {len(codes)}")

        if args.merge:
            new_count, updated_count = merge_with_master(codes)
            print(f"\nMerge results: {new_count} new codes, {updated_count} updated")

        # Print summary
        print("\n" + "=" * 60)
        print("OBD-CODES.COM SCRAPE SUMMARY")
        print("=" * 60)
        print(f"Total codes scraped: {len(codes)}")

        categories = {}
        for code in codes:
            cat = code.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

        print("\nBy category:")
        for cat, count in sorted(categories.items()):
            print(f"  {cat}: {count}")
        print("=" * 60)

    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
