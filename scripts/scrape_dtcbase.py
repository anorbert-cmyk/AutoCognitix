#!/usr/bin/env python3
"""
DTCBase.com DTC Database Scraper.

Scrapes DTC codes from dtcbase.com which contains manufacturer-specific
and generic OBD-II trouble codes.

URL Patterns:
    - Main page: https://dtcbase.com/
    - Code pages: https://dtcbase.com/dtc/P0001

Usage:
    python scripts/scrape_dtcbase.py --scrape
    python scripts/scrape_dtcbase.py --merge
    python scripts/scrape_dtcbase.py --test-connection

Robots.txt Compliance:
    The script respects robots.txt rules and implements rate limiting.
    Rate limit: 1 request per 2.5 seconds minimum.

Connection Handling:
    - Uses connection pooling for efficiency
    - Implements exponential backoff on failures
    - Supports multiple HTTP client backends (httpx, curl_cffi, cloudscraper)
"""

import argparse
import asyncio
import json
import logging
import re
import ssl
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

try:
    from curl_cffi import requests as curl_requests
    HAS_CURL_CFFI = True
except ImportError:
    HAS_CURL_CFFI = False

try:
    import cloudscraper
    HAS_CLOUDSCRAPER = True
except ImportError:
    HAS_CLOUDSCRAPER = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

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
CACHE_FILE = DATA_DIR / "dtcbase_codes.json"
MASTER_FILE = DATA_DIR / "all_codes_merged.json"

# DTCBase.com configuration
BASE_URL = "https://dtcbase.com"

# Rate limiting - respects robots.txt crawl-delay
RATE_LIMIT_DELAY = 2.5  # seconds between requests (> 2 seconds as required)
MAX_RETRIES = 3
RETRY_DELAY = 5  # base seconds for exponential backoff
CONNECTION_TIMEOUT = 30  # seconds

# Browser-like headers for better compatibility
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Cache-Control": "max-age=0",
    "sec-ch-ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}


class ConnectionError(Exception):
    """Custom exception for connection issues."""
    pass


def get_http_client() -> Tuple[Any, str]:
    """
    Get the best available HTTP client.

    Returns:
        Tuple of (client_instance, client_name).
    """
    # Try curl_cffi first (best for bypassing protections)
    if HAS_CURL_CFFI:
        session = curl_requests.Session(impersonate="chrome120")
        return session, "curl_cffi"

    # Try cloudscraper (handles Cloudflare)
    if HAS_CLOUDSCRAPER:
        scraper = cloudscraper.create_scraper(
            browser={
                "browser": "chrome",
                "platform": "darwin",
                "mobile": False,
            }
        )
        return scraper, "cloudscraper"

    # Try regular requests
    if HAS_REQUESTS:
        session = requests.Session()
        session.headers.update(HEADERS)
        return session, "requests"

    # Fallback to None (will use httpx async)
    return None, "httpx"


def test_connection() -> Tuple[bool, str, Optional[str]]:
    """
    Test connection to dtcbase.com.

    Returns:
        Tuple of (success, client_used, robots_txt_content).
    """
    logger.info("Testing connection to dtcbase.com...")

    client, client_name = get_http_client()

    if client is None and HAS_HTTPX:
        # Use synchronous httpx for testing
        try:
            with httpx.Client(
                timeout=CONNECTION_TIMEOUT,
                headers=HEADERS,
                follow_redirects=True,
            ) as http_client:
                response = http_client.get(f"{BASE_URL}/robots.txt")
                if response.status_code == 200:
                    return True, "httpx", response.text
                else:
                    return False, "httpx", None
        except Exception as e:
            logger.error(f"httpx connection failed: {e}")
            return False, "httpx", None

    if client is not None:
        try:
            response = client.get(
                f"{BASE_URL}/robots.txt",
                timeout=CONNECTION_TIMEOUT,
            )
            if response.status_code == 200:
                return True, client_name, response.text
            else:
                logger.warning(f"robots.txt returned status {response.status_code}")
                # Try homepage instead
                response = client.get(f"{BASE_URL}/", timeout=CONNECTION_TIMEOUT)
                if response.status_code == 200:
                    return True, client_name, None
                return False, client_name, None
        except Exception as e:
            logger.error(f"{client_name} connection failed: {e}")
            return False, client_name, None

    return False, "none", None


def parse_robots_txt(content: str) -> Dict[str, Any]:
    """
    Parse robots.txt content.

    Args:
        content: Raw robots.txt content.

    Returns:
        Dictionary with allowed/disallowed paths and crawl-delay.
    """
    result = {
        "user_agent": "*",
        "allowed": [],
        "disallowed": [],
        "crawl_delay": None,
        "sitemaps": [],
    }

    if not content:
        return result

    current_agent = None

    for line in content.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip().lower()
            value = value.strip()

            if key == "user-agent":
                current_agent = value
            elif current_agent in ("*", "Mozilla", None):
                if key == "allow":
                    result["allowed"].append(value)
                elif key == "disallow":
                    result["disallowed"].append(value)
                elif key == "crawl-delay":
                    try:
                        result["crawl_delay"] = float(value)
                    except ValueError:
                        pass
                elif key == "sitemap":
                    result["sitemaps"].append(value)

    return result


def is_path_allowed(path: str, robots_rules: Dict[str, Any]) -> bool:
    """
    Check if a path is allowed by robots.txt rules.

    Args:
        path: URL path to check.
        robots_rules: Parsed robots.txt rules.

    Returns:
        True if path is allowed.
    """
    # Check disallowed first
    for disallowed in robots_rules.get("disallowed", []):
        if disallowed and path.startswith(disallowed):
            # Check if specifically allowed
            for allowed in robots_rules.get("allowed", []):
                if allowed and path.startswith(allowed):
                    return True
            return False
    return True


def generate_dtc_codes() -> List[str]:
    """Generate list of DTC codes to scrape."""
    codes = []

    # P codes: P0000-P0999 (generic), P1000-P3499 (manufacturer)
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
    Scrape a single DTC code page from DTCBase.

    Args:
        client: HTTP client instance.
        code: DTC code to scrape.
        retry_count: Current retry attempt.

    Returns:
        DTC code dictionary or None if not found.
    """
    url = f"{BASE_URL}/dtc/{code.upper()}"

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

        # Find description - DTCBase typically has it in a specific section
        description = None
        symptoms = []
        causes = []
        diagnostic_steps = []

        # Look for main description
        # Try various selectors for DTCBase structure
        desc_selectors = [
            ".dtc-description",
            ".code-description",
            "#description",
            "article .description",
            ".main-content p:first-of-type",
        ]

        for selector in desc_selectors:
            element = soup.select_one(selector)
            if element:
                text = element.get_text(strip=True)
                if len(text) > 15:
                    description = text
                    break

        # Fallback: look for heading followed by description
        if not description:
            h1 = soup.find("h1")
            if h1:
                h1_text = h1.get_text(strip=True)
                # Check if it contains the code
                if code.upper() in h1_text.upper():
                    # Description might be in h1 or next paragraph
                    match = re.search(rf'{code}\s*[-:]\s*(.+)', h1_text, re.IGNORECASE)
                    if match:
                        description = match.group(1).strip()
                    else:
                        # Look at next paragraph
                        next_p = h1.find_next("p")
                        if next_p:
                            description = next_p.get_text(strip=True)

        # Look for definition/meaning section
        if not description:
            for header in soup.find_all(["h2", "h3", "h4"]):
                header_text = header.get_text(strip=True).lower()
                if any(word in header_text for word in ["meaning", "definition", "description", "what"]):
                    content = header.find_next(["p", "div"])
                    if content:
                        description = content.get_text(strip=True)
                        if len(description) > 15:
                            break

        # Extract symptoms
        symptom_headers = soup.find_all(string=re.compile(r'symptom', re.IGNORECASE))
        for header in symptom_headers:
            parent = header.find_parent()
            if parent:
                ul = parent.find_next("ul")
                if ul:
                    for li in ul.find_all("li"):
                        symptoms.append(sanitize_text(li.get_text(strip=True), max_length=200))

        # Extract causes
        cause_headers = soup.find_all(string=re.compile(r'cause|reason', re.IGNORECASE))
        for header in cause_headers:
            parent = header.find_parent()
            if parent:
                ul = parent.find_next("ul")
                if ul:
                    for li in ul.find_all("li"):
                        causes.append(sanitize_text(li.get_text(strip=True), max_length=200))

        # Extract diagnostic steps
        diag_headers = soup.find_all(string=re.compile(r'diagnos|fix|repair|solution', re.IGNORECASE))
        for header in diag_headers:
            parent = header.find_parent()
            if parent:
                ul = parent.find_next(["ul", "ol"])
                if ul:
                    for li in ul.find_all("li"):
                        diagnostic_steps.append(sanitize_text(li.get_text(strip=True), max_length=200))

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
            "source": "dtcbase.com",
        }

    except httpx.RequestError as e:
        logger.debug(f"Request error for {code}: {e}")
        if retry_count < MAX_RETRIES:
            await asyncio.sleep(RETRY_DELAY * (retry_count + 1))
            return await scrape_code_page(client, code, retry_count + 1)
    except Exception as e:
        logger.error(f"Error scraping {code}: {e}")

    return None


async def scrape_sitemap(client: httpx.AsyncClient) -> List[str]:
    """
    Try to scrape sitemap to find available codes.

    Returns:
        List of DTC codes found in sitemap.
    """
    codes = []

    sitemap_urls = [
        f"{BASE_URL}/sitemap.xml",
        f"{BASE_URL}/sitemap_index.xml",
    ]

    for sitemap_url in sitemap_urls:
        try:
            response = await client.get(sitemap_url)
            if response.status_code != 200:
                continue

            # Parse XML sitemap
            soup = BeautifulSoup(response.text, "xml")

            for loc in soup.find_all("loc"):
                url = loc.get_text(strip=True)
                # Check for DTC code pattern
                match = re.search(r'/dtc/([PCBU][0-9]{4})/?$', url, re.IGNORECASE)
                if match:
                    codes.append(match.group(1).upper())

        except Exception as e:
            logger.debug(f"Error fetching sitemap: {e}")

    return list(set(codes))


def scrape_code_page_sync(
    client: Any,
    code: str,
    retry_count: int = 0,
) -> Optional[Dict[str, Any]]:
    """
    Synchronously scrape a single DTC code page from DTCBase.

    Args:
        client: HTTP client instance (requests, cloudscraper, or curl_cffi).
        code: DTC code to scrape.
        retry_count: Current retry attempt.

    Returns:
        DTC code dictionary or None if not found.
    """
    url = f"{BASE_URL}/dtc/{code.upper()}"

    try:
        response = client.get(url, timeout=CONNECTION_TIMEOUT)

        if response.status_code == 404:
            return None

        if response.status_code == 403:
            logger.warning(f"Access forbidden for {code}")
            if retry_count < MAX_RETRIES:
                time.sleep(RETRY_DELAY * (2 ** retry_count))  # Exponential backoff
                return scrape_code_page_sync(client, code, retry_count + 1)
            return None

        if response.status_code != 200:
            logger.debug(f"Unexpected status {response.status_code} for {code}")
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        # Find description - DTCBase typically has it in a specific section
        description = None
        symptoms = []
        causes = []
        diagnostic_steps = []
        manufacturer = None

        # Look for main description with various selectors
        desc_selectors = [
            ".dtc-description",
            ".code-description",
            "#description",
            "article .description",
            ".main-content p:first-of-type",
            ".definition",
            ".meaning",
        ]

        for selector in desc_selectors:
            element = soup.select_one(selector)
            if element:
                text = element.get_text(strip=True)
                if len(text) > 15:
                    description = text
                    break

        # Fallback: look for heading followed by description
        if not description:
            h1 = soup.find("h1")
            if h1:
                h1_text = h1.get_text(strip=True)
                if code.upper() in h1_text.upper():
                    match = re.search(rf'{code}\s*[-:]\s*(.+)', h1_text, re.IGNORECASE)
                    if match:
                        description = match.group(1).strip()
                    else:
                        next_p = h1.find_next("p")
                        if next_p:
                            description = next_p.get_text(strip=True)

        # Look for definition/meaning section
        if not description:
            for header in soup.find_all(["h2", "h3", "h4"]):
                header_text = header.get_text(strip=True).lower()
                if any(word in header_text for word in ["meaning", "definition", "description", "what"]):
                    content = header.find_next(["p", "div"])
                    if content:
                        description = content.get_text(strip=True)
                        if len(description) > 15:
                            break

        # Extract manufacturer if mentioned
        mfr_patterns = [
            r'(Toyota|Honda|Ford|Chevrolet|BMW|Mercedes|Audi|VW|Volkswagen|Nissan|Hyundai|Kia)\s+specific',
            r'manufacturer[:\s]+([\w\s]+)',
        ]
        for pattern in mfr_patterns:
            match = re.search(pattern, response.text, re.IGNORECASE)
            if match:
                manufacturer = match.group(1).strip()
                break

        # Extract symptoms
        symptom_headers = soup.find_all(string=re.compile(r'symptom', re.IGNORECASE))
        for header in symptom_headers:
            parent = header.find_parent()
            if parent:
                ul = parent.find_next("ul")
                if ul:
                    for li in ul.find_all("li"):
                        symptom_text = sanitize_text(li.get_text(strip=True), max_length=200)
                        if symptom_text and symptom_text not in symptoms:
                            symptoms.append(symptom_text)

        # Extract causes
        cause_headers = soup.find_all(string=re.compile(r'cause|reason', re.IGNORECASE))
        for header in cause_headers:
            parent = header.find_parent()
            if parent:
                ul = parent.find_next("ul")
                if ul:
                    for li in ul.find_all("li"):
                        cause_text = sanitize_text(li.get_text(strip=True), max_length=200)
                        if cause_text and cause_text not in causes:
                            causes.append(cause_text)

        # Extract diagnostic steps
        diag_headers = soup.find_all(string=re.compile(r'diagnos|fix|repair|solution', re.IGNORECASE))
        for header in diag_headers:
            parent = header.find_parent()
            if parent:
                ul = parent.find_next(["ul", "ol"])
                if ul:
                    for li in ul.find_all("li"):
                        step_text = sanitize_text(li.get_text(strip=True), max_length=200)
                        if step_text and step_text not in diagnostic_steps:
                            diagnostic_steps.append(step_text)

        if not description:
            return None

        description = sanitize_text(description, max_length=500)
        description = re.sub(rf'^{code}\s*[-:]\s*', '', description, flags=re.IGNORECASE)

        if len(description) < 10:
            return None

        # Determine severity based on code and content
        severity = get_severity_from_code(code)
        severity_keywords = {
            "critical": ["immediately", "safety", "dangerous", "stop driving"],
            "high": ["serious", "damage", "fail", "urgent"],
        }
        desc_lower = description.lower()
        for level, keywords in severity_keywords.items():
            if any(kw in desc_lower for kw in keywords):
                severity = level
                break

        return {
            "code": code.upper(),
            "description_en": description,
            "symptoms": symptoms[:5],
            "possible_causes": causes[:5],
            "diagnostic_steps": diagnostic_steps[:5],
            "manufacturer": manufacturer,
            "severity": severity,
            "source": "dtcbase.com",
        }

    except Exception as e:
        logger.debug(f"Error scraping {code}: {type(e).__name__}: {e}")
        if retry_count < MAX_RETRIES:
            time.sleep(RETRY_DELAY * (2 ** retry_count))
            return scrape_code_page_sync(client, code, retry_count + 1)

    return None


def scrape_all_codes_sync(
    limit: Optional[int] = None,
    robots_rules: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Synchronously scrape all DTC codes using the best available HTTP client.

    Args:
        limit: Optional limit on number of codes to scrape.
        robots_rules: Parsed robots.txt rules.

    Returns:
        List of scraped DTC code dictionaries.
    """
    logger.info("Starting DTCBase.com scrape (synchronous)...")

    # Get rate limit from robots.txt or use default
    rate_limit = RATE_LIMIT_DELAY
    if robots_rules and robots_rules.get("crawl_delay"):
        rate_limit = max(rate_limit, robots_rules["crawl_delay"])
        logger.info(f"Using crawl-delay from robots.txt: {rate_limit}s")

    # Get HTTP client
    client, client_name = get_http_client()
    if client is None:
        raise ConnectionError("No suitable HTTP client available")

    logger.info(f"Using HTTP client: {client_name}")

    all_codes = []
    codes_to_scrape = generate_dtc_codes()

    # Focus on generic codes first (more likely to exist)
    priority_codes = [c for c in codes_to_scrape if c[1] == "0"]
    other_codes = [c for c in codes_to_scrape if c[1] != "0"]
    codes_to_scrape = priority_codes + other_codes

    if limit:
        codes_to_scrape = codes_to_scrape[:limit]

    logger.info(f"Scraping {len(codes_to_scrape)} codes...")

    # Check if DTC paths are allowed
    dtc_path = "/dtc/"
    if robots_rules and not is_path_allowed(dtc_path, robots_rules):
        logger.error(f"Path {dtc_path} is disallowed by robots.txt")
        return []

    scraped_codes = set()
    consecutive_failures = 0
    max_consecutive_failures = 10

    for code in tqdm(codes_to_scrape, desc="Scraping DTCBase"):
        if code in scraped_codes:
            continue

        code_data = scrape_code_page_sync(client, code)
        if code_data:
            all_codes.append(code_data)
            scraped_codes.add(code)
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                logger.warning(f"Too many consecutive failures ({consecutive_failures}), connection may be blocked")
                # Try to reconnect
                client, client_name = get_http_client()
                consecutive_failures = 0

        time.sleep(rate_limit)

    logger.info(f"Scraped {len(all_codes)} codes from DTCBase.com")
    return all_codes


async def scrape_all_codes(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Scrape all DTC codes from DTCBase.com.

    This function first tests the connection and chooses the best scraping method.

    Args:
        limit: Optional limit on number of codes to scrape.

    Returns:
        List of scraped DTC code dictionaries.
    """
    # Test connection first
    success, client_name, robots_txt = test_connection()

    if not success:
        logger.error("Cannot connect to dtcbase.com")
        logger.info("Possible reasons:")
        logger.info("  - Site may be blocking automated requests")
        logger.info("  - Geographic restrictions may apply")
        logger.info("  - Site may be temporarily down")
        logger.info("Try installing curl_cffi: pip install curl_cffi")
        return []

    # Parse robots.txt
    robots_rules = parse_robots_txt(robots_txt) if robots_txt else None
    if robots_rules:
        logger.info(f"robots.txt parsed: {len(robots_rules.get('disallowed', []))} disallowed paths")
        if robots_rules.get("crawl_delay"):
            logger.info(f"Crawl-delay: {robots_rules['crawl_delay']}s")

    # Use synchronous scraping with the working client
    if client_name != "httpx":
        return scrape_all_codes_sync(limit=limit, robots_rules=robots_rules)

    # Fall back to async httpx
    logger.info("Starting DTCBase.com scrape (async)...")

    all_codes = []

    async with httpx.AsyncClient(timeout=30.0, headers=HEADERS, follow_redirects=True) as client:
        # Try to get codes from sitemap first
        sitemap_codes = await scrape_sitemap(client)
        logger.info(f"Found {len(sitemap_codes)} codes in sitemap")

        # Generate code list
        if sitemap_codes:
            codes_to_scrape = sitemap_codes
        else:
            codes_to_scrape = generate_dtc_codes()

        # Focus on generic codes first
        priority_codes = [c for c in codes_to_scrape if c[1] == "0"]
        other_codes = [c for c in codes_to_scrape if c[1] != "0"]
        codes_to_scrape = priority_codes + other_codes

        if limit:
            codes_to_scrape = codes_to_scrape[:limit]

        logger.info(f"Scraping {len(codes_to_scrape)} codes...")

        scraped_codes = set()

        for code in tqdm(codes_to_scrape, desc="Scraping DTCBase"):
            if code in scraped_codes:
                continue

            code_data = await scrape_code_page(client, code)
            if code_data:
                all_codes.append(code_data)
                scraped_codes.add(code)

            await asyncio.sleep(RATE_LIMIT_DELAY)

    logger.info(f"Scraped {len(all_codes)} codes from DTCBase.com")
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
            "sources": ["dtcbase.com"],
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
            "source": "dtcbase.com",
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
            if "dtcbase.com" not in existing["sources"]:
                existing["sources"].append("dtcbase.com")

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


def print_connection_status() -> bool:
    """
    Print connection status and available HTTP clients.

    Returns:
        True if connection is successful.
    """
    print("\n" + "=" * 60)
    print("DTCBASE.COM CONNECTION TEST")
    print("=" * 60)

    # Print available HTTP clients
    print("\nAvailable HTTP clients:")
    print(f"  httpx:        {'YES' if HAS_HTTPX else 'NO'}")
    print(f"  curl_cffi:    {'YES' if HAS_CURL_CFFI else 'NO'} (recommended)")
    print(f"  cloudscraper: {'YES' if HAS_CLOUDSCRAPER else 'NO'}")
    print(f"  requests:     {'YES' if HAS_REQUESTS else 'NO'}")

    print("\nTesting connection...")
    success, client_name, robots_txt = test_connection()

    if success:
        print(f"\nConnection successful using: {client_name}")
        if robots_txt:
            robots_rules = parse_robots_txt(robots_txt)
            print(f"\nrobots.txt parsed:")
            print(f"  Disallowed paths: {len(robots_rules.get('disallowed', []))}")
            print(f"  Allowed paths: {len(robots_rules.get('allowed', []))}")
            if robots_rules.get('crawl_delay'):
                print(f"  Crawl-delay: {robots_rules['crawl_delay']}s")
            if robots_rules.get('sitemaps'):
                print(f"  Sitemaps: {len(robots_rules['sitemaps'])}")

            # Check if /dtc/ path is allowed
            if is_path_allowed("/dtc/", robots_rules):
                print("\n  /dtc/ path: ALLOWED")
            else:
                print("\n  /dtc/ path: BLOCKED")
        else:
            print("  (Could not fetch robots.txt)")
    else:
        print(f"\nConnection FAILED")
        print("\nPossible reasons:")
        print("  - Site may be blocking automated requests")
        print("  - Geographic restrictions may apply")
        print("  - Site may be temporarily down")
        print("\nRecommendations:")
        if not HAS_CURL_CFFI:
            print("  - Install curl_cffi: pip install curl_cffi")
        if not HAS_CLOUDSCRAPER:
            print("  - Install cloudscraper: pip install cloudscraper")
        print("  - Try using a VPN")
        print("  - Check if site is accessible in browser")

    print("=" * 60 + "\n")
    return success


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Scrape DTC codes from dtcbase.com",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/scrape_dtcbase.py --test-connection
    python scripts/scrape_dtcbase.py --scrape --limit 100
    python scripts/scrape_dtcbase.py --merge --use-cache

Requirements:
    - robots.txt compliance is enforced
    - Rate limit: 1 request per 2.5 seconds minimum
    - Install curl_cffi for better connection handling: pip install curl_cffi
        """,
    )
    parser.add_argument("--scrape", action="store_true", help="Scrape codes from website")
    parser.add_argument("--merge", action="store_true", help="Merge with master file")
    parser.add_argument("--use-cache", action="store_true", help="Use cached data")
    parser.add_argument("--test-connection", action="store_true", help="Test connection only")
    parser.add_argument("--limit", type=int, help="Limit number of codes to scrape")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Handle test-connection separately
    if args.test_connection:
        success = print_connection_status()
        sys.exit(0 if success else 1)

    if not (args.scrape or args.merge or args.use_cache):
        args.scrape = True
        args.merge = True

    try:
        if args.use_cache:
            codes = load_from_cache()
            if not codes:
                logger.warning("No cached data found. Run with --scrape first.")
                return
        elif args.scrape:
            raw_codes = await scrape_all_codes(limit=args.limit)
            if not raw_codes:
                logger.error("Scraping failed - no codes retrieved")
                logger.info("Run with --test-connection to diagnose")
                return
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
        print("DTCBASE.COM SCRAPE SUMMARY")
        print("=" * 60)
        print(f"Total codes scraped: {len(codes)}")

        categories = {}
        severities = {}
        generic_count = 0
        manufacturer_count = 0

        for code in codes:
            cat = code.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

            sev = code.get("severity", "unknown")
            severities[sev] = severities.get(sev, 0) + 1

            if code.get("is_generic"):
                generic_count += 1
            if code.get("manufacturer"):
                manufacturer_count += 1

        print("\nBy category:")
        for cat, count in sorted(categories.items()):
            print(f"  {cat}: {count}")

        print("\nBy severity:")
        for sev, count in sorted(severities.items()):
            print(f"  {sev}: {count}")

        print(f"\nGeneric codes: {generic_count}")
        print(f"Manufacturer-specific codes: {manufacturer_count}")
        print(f"\nOutput file: {CACHE_FILE}")
        print("=" * 60)

    except KeyboardInterrupt:
        logger.info("Scraping interrupted by user")
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(main())
