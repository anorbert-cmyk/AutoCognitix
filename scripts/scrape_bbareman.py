#!/usr/bin/env python3
"""
BBA-Reman.com DTC Database Scraper.

Scrapes DTC codes from bba-reman.com knowledge base which has
professional-grade diagnostic trouble code information.

URL Pattern:
    - Main page: https://www.bba-reman.com/en-us/knowledge-base/dtc-database
    - Code pages: https://www.bba-reman.com/en-us/knowledge-base/dtc-database/{code}

Usage:
    python scripts/scrape_bbareman.py --scrape
    python scripts/scrape_bbareman.py --merge
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
CACHE_FILE = DATA_DIR / "bbareman_codes.json"
MASTER_FILE = DATA_DIR / "all_codes_merged.json"

# BBA-Reman.com configuration
BASE_URL = "https://www.bba-reman.com"
DTC_DATABASE_URL = f"{BASE_URL}/en-us/knowledge-base/dtc-database"

# Rate limiting
RATE_LIMIT_DELAY = 2.5  # seconds between requests
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


def generate_dtc_codes() -> List[str]:
    """Generate list of DTC codes to scrape."""
    codes = []

    # P codes - most common
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


async def scrape_database_index(client: httpx.AsyncClient) -> List[str]:
    """
    Scrape the main DTC database page to find available codes.

    Returns:
        List of DTC codes found.
    """
    codes = []

    try:
        response = await client.get(DTC_DATABASE_URL)
        if response.status_code != 200:
            logger.warning(f"Failed to fetch database index: {response.status_code}")
            return codes

        soup = BeautifulSoup(response.text, "html.parser")

        # Find all links that look like DTC codes
        for a in soup.find_all("a"):
            href = a.get("href", "")
            text = a.get_text(strip=True)

            # Check for DTC code pattern in href or text
            match = re.search(r'([PCBU][0-9]{4})', href.upper())
            if match:
                codes.append(match.group(1))
                continue

            match = re.search(r'([PCBU][0-9]{4})', text.upper())
            if match:
                codes.append(match.group(1))

        # Also look for code tables
        for table in soup.find_all("table"):
            for row in table.find_all("tr"):
                cells = row.find_all(["td", "th"])
                for cell in cells:
                    text = cell.get_text(strip=True).upper()
                    match = re.match(r'^([PCBU][0-9]{4})$', text)
                    if match:
                        codes.append(match.group(1))

        # Look for codes in any list
        for li in soup.find_all("li"):
            text = li.get_text(strip=True).upper()
            match = re.match(r'^([PCBU][0-9]{4})\s', text)
            if match:
                codes.append(match.group(1))

    except Exception as e:
        logger.error(f"Error fetching database index: {e}")

    unique_codes = list(set(codes))
    logger.info(f"Found {len(unique_codes)} codes in database index")
    return unique_codes


async def scrape_code_page(
    client: httpx.AsyncClient,
    code: str,
    retry_count: int = 0,
) -> Optional[Dict[str, Any]]:
    """
    Scrape a single DTC code page from BBA-Reman.

    Args:
        client: HTTP client instance.
        code: DTC code to scrape.
        retry_count: Current retry attempt.

    Returns:
        DTC code dictionary or None if not found.
    """
    # BBA-Reman may use different URL patterns
    url_patterns = [
        f"{DTC_DATABASE_URL}/{code.lower()}",
        f"{DTC_DATABASE_URL}/{code.upper()}",
        f"{BASE_URL}/en-us/knowledge-base/dtc/{code.lower()}",
    ]

    for url in url_patterns:
        try:
            response = await client.get(url)

            if response.status_code == 404:
                continue

            if response.status_code == 403:
                logger.warning(f"Access forbidden for {code}")
                if retry_count < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY * (retry_count + 1))
                    return await scrape_code_page(client, code, retry_count + 1)
                return None

            if response.status_code != 200:
                continue

            soup = BeautifulSoup(response.text, "html.parser")

            description = None
            symptoms = []
            causes = []
            diagnostic_steps = []

            # BBA-Reman has professional-grade content
            # Look for main content area
            content = soup.find("article") or soup.find("main") or soup.find("div", class_="content")

            if content:
                # Find description
                for header in content.find_all(["h1", "h2", "h3"]):
                    header_text = header.get_text(strip=True)
                    # Check if header contains the code
                    if code.upper() in header_text.upper():
                        match = re.search(rf'{code}\s*[-:]\s*(.+)', header_text, re.IGNORECASE)
                        if match:
                            description = match.group(1).strip()
                            break

                # Look for definition section
                if not description:
                    for header in content.find_all(["h2", "h3", "h4"]):
                        header_text = header.get_text(strip=True).lower()
                        if any(word in header_text for word in ["definition", "meaning", "description", "what"]):
                            next_elem = header.find_next(["p", "div"])
                            if next_elem:
                                description = next_elem.get_text(strip=True)
                                if len(description) > 15:
                                    break

                # Find symptoms
                for header in content.find_all(["h2", "h3", "h4"]):
                    header_text = header.get_text(strip=True).lower()
                    if "symptom" in header_text:
                        ul = header.find_next("ul")
                        if ul:
                            for li in ul.find_all("li"):
                                symptoms.append(sanitize_text(li.get_text(strip=True), max_length=200))

                # Find causes
                for header in content.find_all(["h2", "h3", "h4"]):
                    header_text = header.get_text(strip=True).lower()
                    if any(word in header_text for word in ["cause", "reason", "trigger"]):
                        ul = header.find_next("ul")
                        if ul:
                            for li in ul.find_all("li"):
                                causes.append(sanitize_text(li.get_text(strip=True), max_length=200))

                # Find diagnostic/repair steps
                for header in content.find_all(["h2", "h3", "h4"]):
                    header_text = header.get_text(strip=True).lower()
                    if any(word in header_text for word in ["diagnos", "repair", "fix", "solution", "troubleshoot"]):
                        ul = header.find_next(["ul", "ol"])
                        if ul:
                            for li in ul.find_all("li"):
                                diagnostic_steps.append(sanitize_text(li.get_text(strip=True), max_length=200))

            # Fallback: check title
            if not description:
                title = soup.find("title")
                if title:
                    title_text = title.get_text(strip=True)
                    match = re.search(rf'{code}\s*[-:]\s*(.+)', title_text, re.IGNORECASE)
                    if match:
                        description = match.group(1).strip()

            # Fallback: first paragraph with code mention
            if not description:
                for p in soup.find_all("p"):
                    text = p.get_text(strip=True)
                    if code.upper() in text.upper() and len(text) > 20:
                        description = text
                        break

            if not description:
                continue

            description = sanitize_text(description, max_length=500)
            description = re.sub(rf'^{code}\s*[-:]\s*', '', description, flags=re.IGNORECASE)

            if len(description) < 10:
                continue

            return {
                "code": code.upper(),
                "description_en": description,
                "symptoms": symptoms[:5],
                "possible_causes": causes[:5],
                "diagnostic_steps": diagnostic_steps[:5],
                "source": "bba-reman.com",
            }

        except httpx.RequestError as e:
            logger.debug(f"Request error for {code}: {e}")
            if retry_count < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY * (retry_count + 1))
                return await scrape_code_page(client, code, retry_count + 1)
        except Exception as e:
            logger.error(f"Error scraping {code}: {e}")

    return None


async def scrape_all_codes(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Scrape all DTC codes from BBA-Reman.com.

    Args:
        limit: Optional limit on number of codes to scrape.

    Returns:
        List of scraped DTC code dictionaries.
    """
    logger.info("Starting BBA-Reman.com scrape...")

    all_codes = []

    async with httpx.AsyncClient(timeout=30.0, headers=HEADERS, follow_redirects=True) as client:
        # First try to get codes from the database index
        index_codes = await scrape_database_index(client)

        if index_codes:
            codes_to_scrape = index_codes
        else:
            # Fall back to generated codes
            codes_to_scrape = generate_dtc_codes()

        # Focus on generic codes first
        priority_codes = [c for c in codes_to_scrape if c[1] == "0"]
        other_codes = [c for c in codes_to_scrape if c[1] != "0"]
        codes_to_scrape = priority_codes + other_codes

        if limit:
            codes_to_scrape = codes_to_scrape[:limit]

        logger.info(f"Scraping {len(codes_to_scrape)} codes...")

        scraped = set()

        for code in tqdm(codes_to_scrape, desc="Scraping BBA-Reman"):
            if code in scraped:
                continue

            code_data = await scrape_code_page(client, code)
            if code_data:
                all_codes.append(code_data)
                scraped.add(code)

            await asyncio.sleep(RATE_LIMIT_DELAY)

    logger.info(f"Scraped {len(all_codes)} codes from BBA-Reman.com")
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
            "sources": ["bba-reman.com"],
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
            "source": "bba-reman.com",
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
            if "bba-reman.com" not in existing["sources"]:
                existing["sources"].append("bba-reman.com")

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
    parser = argparse.ArgumentParser(description="Scrape DTC codes from bba-reman.com")
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
        print("BBA-REMAN.COM SCRAPE SUMMARY")
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
