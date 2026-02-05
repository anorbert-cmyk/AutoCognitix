#!/usr/bin/env python3
"""
TroubleCodes.net DTC Database Scraper.

Scrapes DTC codes from troublecodes.net which has a comprehensive
database of OBD-II trouble codes.

URL Patterns:
    - Main page: https://www.troublecodes.net/
    - Code pages: https://www.troublecodes.net/pcodes/p0001/

Usage:
    python scripts/scrape_troublecodes.py --scrape
    python scripts/scrape_troublecodes.py --merge
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
CACHE_FILE = DATA_DIR / "troublecodes_net.json"
MASTER_FILE = DATA_DIR / "all_codes_merged.json"

# TroubleCodes.net configuration
BASE_URL = "https://www.troublecodes.net"

# Rate limiting
RATE_LIMIT_DELAY = 2.0  # seconds between requests
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# User agent
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

# URL patterns for different code categories
CODE_URL_PATTERNS = {
    "P": "pcodes",
    "B": "bcodes",
    "C": "ccodes",
    "U": "ucodes",
}


def generate_dtc_codes() -> List[str]:
    """Generate list of DTC codes to scrape."""
    codes = []

    # P codes
    for i in range(0, 1000):
        codes.append(f"P{i:04d}")
    for i in range(1000, 3500):
        codes.append(f"P{i:04d}")

    # B codes
    for i in range(0, 1000):
        codes.append(f"B{i:04d}")
    for i in range(1000, 3500):
        codes.append(f"B{i:04d}")

    # C codes
    for i in range(0, 1000):
        codes.append(f"C{i:04d}")
    for i in range(1000, 3500):
        codes.append(f"C{i:04d}")

    # U codes
    for i in range(0, 500):
        codes.append(f"U{i:04d}")
    for i in range(1000, 3200):
        codes.append(f"U{i:04d}")

    return codes


async def scrape_code_page(
    client: httpx.AsyncClient,
    code: str,
    retry_count: int = 0,
) -> Optional[Dict[str, Any]]:
    """
    Scrape a single DTC code page from TroubleCodes.net.

    Args:
        client: HTTP client instance.
        code: DTC code to scrape.
        retry_count: Current retry attempt.

    Returns:
        DTC code dictionary or None if not found.
    """
    prefix = code[0].upper()
    url_prefix = CODE_URL_PATTERNS.get(prefix, "pcodes")
    url = f"{BASE_URL}/{url_prefix}/{code.lower()}/"

    try:
        response = await client.get(url)

        if response.status_code == 404:
            return None

        if response.status_code == 403:
            logger.warning(f"Access forbidden for {code}")
            if retry_count < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY * (retry_count + 1))
                return await scrape_code_page(client, code, retry_count + 1)
            return None

        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        description = None
        symptoms = []
        causes = []
        diagnostic_steps = []

        # TroubleCodes.net typically has article structure
        # Look for main content area
        article = soup.find("article") or soup.find("main") or soup.find("div", class_="content")

        if article:
            # Get description from first paragraph or heading
            h1 = article.find("h1")
            if h1:
                h1_text = h1.get_text(strip=True)
                match = re.search(rf'{code}\s*[-:]\s*(.+)', h1_text, re.IGNORECASE)
                if match:
                    description = match.group(1).strip()

            if not description:
                # Try to find first meaningful paragraph
                for p in article.find_all("p"):
                    text = p.get_text(strip=True)
                    if len(text) > 20 and not text.lower().startswith(("advertisement", "sponsored")):
                        description = text
                        break

            # Find symptoms
            symptom_section = article.find(string=re.compile(r'symptom', re.IGNORECASE))
            if symptom_section:
                parent = symptom_section.find_parent()
                if parent:
                    ul = parent.find_next("ul")
                    if ul:
                        for li in ul.find_all("li"):
                            symptoms.append(sanitize_text(li.get_text(strip=True), max_length=200))

            # Find causes
            cause_section = article.find(string=re.compile(r'cause|possible', re.IGNORECASE))
            if cause_section:
                parent = cause_section.find_parent()
                if parent:
                    ul = parent.find_next("ul")
                    if ul:
                        for li in ul.find_all("li"):
                            causes.append(sanitize_text(li.get_text(strip=True), max_length=200))

            # Find diagnostic steps
            diag_section = article.find(string=re.compile(r'diagnos|troubleshoot|repair', re.IGNORECASE))
            if diag_section:
                parent = diag_section.find_parent()
                if parent:
                    ul = parent.find_next(["ul", "ol"])
                    if ul:
                        for li in ul.find_all("li"):
                            diagnostic_steps.append(sanitize_text(li.get_text(strip=True), max_length=200))

        # Fallback if no article found
        if not description:
            h1 = soup.find("h1")
            if h1:
                h1_text = h1.get_text(strip=True)
                match = re.search(rf'{code}\s*[-:]\s*(.+)', h1_text, re.IGNORECASE)
                if match:
                    description = match.group(1).strip()

        if not description:
            title = soup.find("title")
            if title:
                title_text = title.get_text(strip=True)
                match = re.search(rf'{code}\s*[-:]\s*(.+)', title_text, re.IGNORECASE)
                if match:
                    description = match.group(1).strip()

        if not description:
            return None

        description = sanitize_text(description, max_length=500)
        description = re.sub(rf'^{code}\s*[-:]\s*', '', description, flags=re.IGNORECASE)

        if len(description) < 10:
            return None

        return {
            "code": code.upper(),
            "description_en": description,
            "symptoms": symptoms[:5],
            "possible_causes": causes[:5],
            "diagnostic_steps": diagnostic_steps[:5],
            "source": "troublecodes.net",
        }

    except httpx.RequestError as e:
        logger.debug(f"Request error for {code}: {e}")
        if retry_count < MAX_RETRIES:
            await asyncio.sleep(RETRY_DELAY * (retry_count + 1))
            return await scrape_code_page(client, code, retry_count + 1)
    except Exception as e:
        logger.error(f"Error scraping {code}: {e}")

    return None


async def scrape_index_pages(client: httpx.AsyncClient) -> List[Dict[str, Any]]:
    """
    Scrape index/listing pages to find codes with descriptions.

    Returns:
        List of DTC code dictionaries.
    """
    codes = []

    # TroubleCodes.net has index pages for each category
    index_urls = [
        f"{BASE_URL}/pcodes/",
        f"{BASE_URL}/bcodes/",
        f"{BASE_URL}/ccodes/",
        f"{BASE_URL}/ucodes/",
    ]

    for url in index_urls:
        try:
            response = await client.get(url)
            if response.status_code != 200:
                continue

            soup = BeautifulSoup(response.text, "html.parser")

            # Find all links to code pages
            for a in soup.find_all("a"):
                href = a.get("href", "")
                text = a.get_text(strip=True)

                # Check for DTC code pattern in href
                match = re.search(r'/([PCBU][0-9]{4})/?$', href, re.IGNORECASE)
                if match:
                    code = match.group(1).upper()

                    # Try to extract description
                    description = ""

                    # Check parent element for description
                    parent = a.find_parent(["li", "td", "div", "tr"])
                    if parent:
                        full_text = parent.get_text(strip=True)
                        # Remove the code from the text
                        description = re.sub(rf'^{code}\s*[-:]\s*', '', full_text, flags=re.IGNORECASE)
                        description = description.strip()

                    if description and len(description) > 15:
                        codes.append({
                            "code": code,
                            "description_en": sanitize_text(description, max_length=500),
                            "source": "troublecodes.net",
                        })

            await asyncio.sleep(RATE_LIMIT_DELAY)

        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")

    return codes


async def scrape_all_codes(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Scrape all DTC codes from TroubleCodes.net.

    Args:
        limit: Optional limit on number of codes to scrape.

    Returns:
        List of scraped DTC code dictionaries.
    """
    logger.info("Starting TroubleCodes.net scrape...")

    all_codes = []

    async with httpx.AsyncClient(timeout=30.0, headers=HEADERS, follow_redirects=True) as client:
        # First scrape index pages for bulk data
        index_codes = await scrape_index_pages(client)
        logger.info(f"Got {len(index_codes)} codes from index pages")
        all_codes.extend(index_codes)

        existing_codes = {c["code"] for c in all_codes}

        # Generate remaining codes to scrape
        codes_to_scrape = generate_dtc_codes()
        codes_to_scrape = [c for c in codes_to_scrape if c not in existing_codes]

        # Focus on generic codes
        priority_codes = [c for c in codes_to_scrape if c[1] == "0"]

        if limit:
            priority_codes = priority_codes[:limit]

        logger.info(f"Scraping {len(priority_codes)} individual code pages...")

        for code in tqdm(priority_codes, desc="Scraping TroubleCodes.net"):
            if code in existing_codes:
                continue

            code_data = await scrape_code_page(client, code)
            if code_data:
                all_codes.append(code_data)
                existing_codes.add(code)

            await asyncio.sleep(RATE_LIMIT_DELAY)

    logger.info(f"Scraped {len(all_codes)} codes from TroubleCodes.net")
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
            "diagnostic_steps": code_data.get("diagnostic_steps", []),
            "related_codes": [],
            "sources": ["troublecodes.net"],
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
            "source": "troublecodes.net",
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

    return data.get("codes", [])


def merge_with_master(new_codes: List[Dict[str, Any]]) -> tuple[int, int]:
    """Merge scraped codes with master file."""
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

            if "sources" not in existing:
                existing["sources"] = []
            if "troublecodes.net" not in existing["sources"]:
                existing["sources"].append("troublecodes.net")

            if len(code_data.get("description_en", "")) > len(existing.get("description_en", "")):
                existing["description_en"] = code_data["description_en"]
                updated_count += 1

            for field in ["symptoms", "possible_causes", "diagnostic_steps"]:
                if code_data.get(field):
                    existing_items = set(existing.get(field, []))
                    for item in code_data[field]:
                        if item not in existing_items:
                            existing.setdefault(field, []).append(item)
        else:
            master_codes[code] = code_data
            new_count += 1

    master_data["codes"] = sorted(master_codes.values(), key=lambda x: x["code"])
    master_data["metadata"]["total_codes"] = len(master_data["codes"])
    master_data["metadata"]["generated_at"] = datetime.now(timezone.utc).isoformat()

    with open(MASTER_FILE, "w", encoding="utf-8") as f:
        json.dump(master_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Merged: {new_count} new, {updated_count} updated")
    return new_count, updated_count


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Scrape DTC codes from troublecodes.net")
    parser.add_argument("--scrape", action="store_true", help="Scrape codes from website")
    parser.add_argument("--merge", action="store_true", help="Merge with master file")
    parser.add_argument("--use-cache", action="store_true", help="Use cached data")
    parser.add_argument("--limit", type=int, help="Limit number of codes to scrape")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

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

        print("\n" + "=" * 60)
        print("TROUBLECODES.NET SCRAPE SUMMARY")
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
