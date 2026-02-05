#!/usr/bin/env python3
"""
RepairPal Repair Cost and Procedure Scraper.

Scrapes repair cost estimates, procedures, and related DTC codes from RepairPal.
Uses Playwright to handle Cloudflare protection.

URL Patterns:
    - Repair estimates: https://repairpal.com/estimator/[make]/[model]/[repair-type]
    - Common repairs: https://repairpal.com/[repair-type]-cost

Usage:
    python scripts/scrape_repairpal.py --scrape
    python scripts/scrape_repairpal.py --merge
    python scripts/scrape_repairpal.py --scrape --limit 50

Requirements:
    pip install playwright
    playwright install chromium
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

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils import (
    sanitize_text,
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
DATA_DIR = PROJECT_ROOT / "data" / "repairs"
CACHE_FILE = DATA_DIR / "repairpal_data.json"
DTC_CODES_FILE = PROJECT_ROOT / "data" / "dtc_codes" / "all_codes_merged.json"

# RepairPal configuration
BASE_URL = "https://repairpal.com"

# Rate limiting (respecting robots.txt and server load)
RATE_LIMIT_DELAY = 2.0  # seconds between requests
PAGE_LOAD_TIMEOUT = 30000  # milliseconds
MAX_RETRIES = 3

# Common repair types to scrape
COMMON_REPAIRS = [
    "oxygen-sensor-replacement",
    "catalytic-converter-replacement",
    "ignition-coil-replacement",
    "spark-plug-replacement",
    "mass-air-flow-sensor-replacement",
    "throttle-body-replacement",
    "fuel-pump-replacement",
    "fuel-injector-replacement",
    "alternator-replacement",
    "starter-motor-replacement",
    "battery-replacement",
    "timing-belt-replacement",
    "timing-chain-replacement",
    "water-pump-replacement",
    "thermostat-replacement",
    "radiator-replacement",
    "brake-pad-replacement",
    "brake-rotor-replacement",
    "brake-caliper-replacement",
    "wheel-bearing-replacement",
    "cv-axle-replacement",
    "tie-rod-end-replacement",
    "ball-joint-replacement",
    "control-arm-replacement",
    "strut-replacement",
    "shock-absorber-replacement",
    "ac-compressor-replacement",
    "heater-core-replacement",
    "evaporator-replacement",
    "power-steering-pump-replacement",
    "transmission-fluid-change",
    "transmission-replacement",
    "clutch-replacement",
    "head-gasket-replacement",
    "engine-replacement",
    "egr-valve-replacement",
    "pcv-valve-replacement",
    "camshaft-position-sensor-replacement",
    "crankshaft-position-sensor-replacement",
    "knock-sensor-replacement",
    "coolant-temperature-sensor-replacement",
    "oil-pressure-sensor-replacement",
    "map-sensor-replacement",
    "idle-air-control-valve-replacement",
    "evap-canister-replacement",
    "purge-valve-replacement",
    "abs-sensor-replacement",
    "abs-module-replacement",
    "airbag-sensor-replacement",
]

# DTC code to repair type mapping
DTC_REPAIR_MAPPING = {
    # Oxygen sensor related
    "P0130": ["oxygen-sensor-replacement"],
    "P0131": ["oxygen-sensor-replacement"],
    "P0132": ["oxygen-sensor-replacement"],
    "P0133": ["oxygen-sensor-replacement"],
    "P0134": ["oxygen-sensor-replacement"],
    "P0135": ["oxygen-sensor-replacement"],
    "P0136": ["oxygen-sensor-replacement"],
    "P0137": ["oxygen-sensor-replacement"],
    "P0138": ["oxygen-sensor-replacement"],
    "P0139": ["oxygen-sensor-replacement"],
    "P0140": ["oxygen-sensor-replacement"],
    "P0141": ["oxygen-sensor-replacement"],
    # Catalytic converter
    "P0420": ["catalytic-converter-replacement"],
    "P0421": ["catalytic-converter-replacement"],
    "P0430": ["catalytic-converter-replacement"],
    "P0431": ["catalytic-converter-replacement"],
    # Ignition/Misfire
    "P0300": ["ignition-coil-replacement", "spark-plug-replacement"],
    "P0301": ["ignition-coil-replacement", "spark-plug-replacement"],
    "P0302": ["ignition-coil-replacement", "spark-plug-replacement"],
    "P0303": ["ignition-coil-replacement", "spark-plug-replacement"],
    "P0304": ["ignition-coil-replacement", "spark-plug-replacement"],
    "P0305": ["ignition-coil-replacement", "spark-plug-replacement"],
    "P0306": ["ignition-coil-replacement", "spark-plug-replacement"],
    "P0307": ["ignition-coil-replacement", "spark-plug-replacement"],
    "P0308": ["ignition-coil-replacement", "spark-plug-replacement"],
    # MAF sensor
    "P0100": ["mass-air-flow-sensor-replacement"],
    "P0101": ["mass-air-flow-sensor-replacement"],
    "P0102": ["mass-air-flow-sensor-replacement"],
    "P0103": ["mass-air-flow-sensor-replacement"],
    "P0104": ["mass-air-flow-sensor-replacement"],
    # Throttle body
    "P0120": ["throttle-body-replacement"],
    "P0121": ["throttle-body-replacement"],
    "P0122": ["throttle-body-replacement"],
    "P0123": ["throttle-body-replacement"],
    "P0124": ["throttle-body-replacement"],
    "P2135": ["throttle-body-replacement"],
    # Fuel system
    "P0230": ["fuel-pump-replacement"],
    "P0231": ["fuel-pump-replacement"],
    "P0232": ["fuel-pump-replacement"],
    "P0201": ["fuel-injector-replacement"],
    "P0202": ["fuel-injector-replacement"],
    "P0203": ["fuel-injector-replacement"],
    "P0204": ["fuel-injector-replacement"],
    # Cooling system
    "P0115": ["coolant-temperature-sensor-replacement"],
    "P0116": ["coolant-temperature-sensor-replacement", "thermostat-replacement"],
    "P0117": ["coolant-temperature-sensor-replacement"],
    "P0118": ["coolant-temperature-sensor-replacement"],
    "P0125": ["thermostat-replacement"],
    "P0128": ["thermostat-replacement"],
    # Camshaft/Crankshaft
    "P0340": ["camshaft-position-sensor-replacement"],
    "P0341": ["camshaft-position-sensor-replacement"],
    "P0342": ["camshaft-position-sensor-replacement"],
    "P0343": ["camshaft-position-sensor-replacement"],
    "P0335": ["crankshaft-position-sensor-replacement"],
    "P0336": ["crankshaft-position-sensor-replacement"],
    "P0337": ["crankshaft-position-sensor-replacement"],
    "P0338": ["crankshaft-position-sensor-replacement"],
    # Knock sensor
    "P0325": ["knock-sensor-replacement"],
    "P0326": ["knock-sensor-replacement"],
    "P0327": ["knock-sensor-replacement"],
    "P0328": ["knock-sensor-replacement"],
    # EVAP system
    "P0440": ["evap-canister-replacement", "purge-valve-replacement"],
    "P0441": ["purge-valve-replacement"],
    "P0442": ["evap-canister-replacement"],
    "P0443": ["purge-valve-replacement"],
    "P0446": ["evap-canister-replacement"],
    "P0455": ["evap-canister-replacement"],
    # EGR
    "P0400": ["egr-valve-replacement"],
    "P0401": ["egr-valve-replacement"],
    "P0402": ["egr-valve-replacement"],
    "P0403": ["egr-valve-replacement"],
    "P0404": ["egr-valve-replacement"],
    # MAP sensor
    "P0105": ["map-sensor-replacement"],
    "P0106": ["map-sensor-replacement"],
    "P0107": ["map-sensor-replacement"],
    "P0108": ["map-sensor-replacement"],
    # Idle control
    "P0505": ["idle-air-control-valve-replacement"],
    "P0506": ["idle-air-control-valve-replacement"],
    "P0507": ["idle-air-control-valve-replacement"],
    # ABS
    "C0035": ["abs-sensor-replacement"],
    "C0040": ["abs-sensor-replacement"],
    "C0045": ["abs-sensor-replacement"],
    "C0050": ["abs-sensor-replacement"],
    "C0265": ["abs-module-replacement"],
    "C0266": ["abs-module-replacement"],
    # Airbag
    "B0001": ["airbag-sensor-replacement"],
    "B0002": ["airbag-sensor-replacement"],
    "B0100": ["airbag-sensor-replacement"],
}


async def check_robots_txt() -> Dict[str, Any]:
    """
    Check robots.txt for scraping permissions.

    Returns:
        Dictionary with allowed/disallowed paths and crawl delay.
    """
    try:
        from playwright.async_api import async_playwright

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            await page.goto(f"{BASE_URL}/robots.txt", timeout=PAGE_LOAD_TIMEOUT)
            content = await page.content()
            await browser.close()

            # Parse robots.txt
            result = {
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "disallowed": [],
                "allowed": [],
                "crawl_delay": None,
                "raw_content": content[:2000] if content else None,
            }

            # Look for disallowed paths
            for line in content.split('\n'):
                line = line.strip().lower()
                if line.startswith('disallow:'):
                    path = line.replace('disallow:', '').strip()
                    if path:
                        result["disallowed"].append(path)
                elif line.startswith('allow:'):
                    path = line.replace('allow:', '').strip()
                    if path:
                        result["allowed"].append(path)
                elif line.startswith('crawl-delay:'):
                    try:
                        delay = float(line.replace('crawl-delay:', '').strip())
                        result["crawl_delay"] = delay
                    except ValueError:
                        pass

            return result

    except Exception as e:
        logger.error(f"Error checking robots.txt: {e}")
        return {"error": str(e)}


async def scrape_repair_page(
    page,
    repair_type: str,
    retry_count: int = 0,
) -> Optional[Dict[str, Any]]:
    """
    Scrape a single repair type page from RepairPal.

    Args:
        page: Playwright page instance.
        repair_type: Repair type slug (e.g., "oxygen-sensor-replacement").
        retry_count: Current retry attempt.

    Returns:
        Repair data dictionary or None if not found.
    """
    url = f"{BASE_URL}/{repair_type}-cost"

    try:
        response = await page.goto(url, timeout=PAGE_LOAD_TIMEOUT, wait_until="networkidle")

        if response.status == 404:
            logger.debug(f"Page not found: {url}")
            return None

        if response.status == 403:
            logger.warning(f"Access forbidden for {repair_type}")
            if retry_count < MAX_RETRIES:
                await asyncio.sleep(RATE_LIMIT_DELAY * (retry_count + 1))
                return await scrape_repair_page(page, repair_type, retry_count + 1)
            return None

        if response.status != 200:
            logger.debug(f"Non-200 status for {url}: {response.status}")
            return None

        # Wait for content to load
        await page.wait_for_load_state("domcontentloaded")
        await asyncio.sleep(1)  # Additional wait for dynamic content

        # Extract repair data
        repair_data = {
            "repair_type": repair_type,
            "url": url,
            "scraped_at": datetime.now(timezone.utc).isoformat(),
        }

        # Try to get title/repair name
        title_elem = await page.query_selector("h1")
        if title_elem:
            repair_data["name"] = sanitize_text(await title_elem.inner_text())
        else:
            # Fallback: format repair type
            repair_data["name"] = repair_type.replace("-", " ").title()

        # Try to get cost estimate
        cost_data = await extract_cost_data(page)
        if cost_data:
            repair_data.update(cost_data)

        # Try to get repair description
        description = await extract_description(page)
        if description:
            repair_data["description"] = description

        # Try to get time estimate
        time_estimate = await extract_time_estimate(page)
        if time_estimate:
            repair_data["time_estimate"] = time_estimate

        # Try to get repair procedure steps
        procedure_steps = await extract_procedure_steps(page)
        if procedure_steps:
            repair_data["procedure_steps"] = procedure_steps

        # Try to get related symptoms
        symptoms = await extract_symptoms(page)
        if symptoms:
            repair_data["symptoms"] = symptoms

        # Try to get affected components
        components = await extract_components(page)
        if components:
            repair_data["affected_components"] = components

        # Try to get vehicle compatibility
        vehicles = await extract_vehicles(page)
        if vehicles:
            repair_data["affected_vehicles"] = vehicles

        # Map to DTC codes
        related_dtcs = get_related_dtc_codes(repair_type)
        if related_dtcs:
            repair_data["related_dtc_codes"] = related_dtcs

        # Validate we got meaningful data
        if not repair_data.get("description") and not repair_data.get("cost_low"):
            logger.debug(f"No meaningful data extracted for {repair_type}")
            return None

        return repair_data

    except Exception as e:
        logger.error(f"Error scraping {repair_type}: {e}")
        if retry_count < MAX_RETRIES:
            await asyncio.sleep(RATE_LIMIT_DELAY * (retry_count + 1))
            return await scrape_repair_page(page, repair_type, retry_count + 1)
        return None


async def extract_cost_data(page) -> Optional[Dict[str, Any]]:
    """Extract cost estimate data from page."""
    cost_data = {}

    try:
        # Look for cost range elements (common patterns on repair sites)
        selectors = [
            "[class*='cost']",
            "[class*='price']",
            "[class*='estimate']",
            "[data-cost]",
            ".repair-cost",
            ".price-range",
        ]

        for selector in selectors:
            elements = await page.query_selector_all(selector)
            for elem in elements:
                text = await elem.inner_text()

                # Extract dollar amounts
                amounts = re.findall(r'\$[\d,]+(?:\.\d{2})?', text)
                if len(amounts) >= 2:
                    cost_data["cost_low"] = parse_price(amounts[0])
                    cost_data["cost_high"] = parse_price(amounts[1])
                    break
                elif len(amounts) == 1:
                    cost_data["cost_average"] = parse_price(amounts[0])

        # Try to extract labor vs parts breakdown
        labor_elem = await page.query_selector("[class*='labor']")
        if labor_elem:
            text = await labor_elem.inner_text()
            amounts = re.findall(r'\$[\d,]+(?:\.\d{2})?', text)
            if amounts:
                cost_data["labor_cost"] = parse_price(amounts[0])

        parts_elem = await page.query_selector("[class*='parts']")
        if parts_elem:
            text = await parts_elem.inner_text()
            amounts = re.findall(r'\$[\d,]+(?:\.\d{2})?', text)
            if amounts:
                cost_data["parts_cost"] = parse_price(amounts[0])

        return cost_data if cost_data else None

    except Exception as e:
        logger.debug(f"Error extracting cost data: {e}")
        return None


async def extract_description(page) -> Optional[str]:
    """Extract repair description from page."""
    try:
        # Common description selectors
        selectors = [
            ".repair-description",
            ".service-description",
            "[class*='description']",
            "article p",
            ".content p:first-of-type",
            "main p:first-of-type",
        ]

        for selector in selectors:
            elem = await page.query_selector(selector)
            if elem:
                text = await elem.inner_text()
                if len(text) > 50:  # Meaningful description
                    return sanitize_text(text, max_length=1000)

        return None

    except Exception as e:
        logger.debug(f"Error extracting description: {e}")
        return None


async def extract_time_estimate(page) -> Optional[Dict[str, Any]]:
    """Extract time estimate from page."""
    try:
        selectors = [
            "[class*='time']",
            "[class*='duration']",
            "[class*='hours']",
        ]

        for selector in selectors:
            elements = await page.query_selector_all(selector)
            for elem in elements:
                text = await elem.inner_text()

                # Look for hour patterns
                hours_match = re.search(r'(\d+(?:\.\d+)?)\s*-?\s*(\d+(?:\.\d+)?)?\s*hours?', text, re.IGNORECASE)
                if hours_match:
                    result = {"hours_low": float(hours_match.group(1))}
                    if hours_match.group(2):
                        result["hours_high"] = float(hours_match.group(2))
                    return result

                # Look for minute patterns
                mins_match = re.search(r'(\d+)\s*-?\s*(\d+)?\s*min', text, re.IGNORECASE)
                if mins_match:
                    result = {"minutes_low": int(mins_match.group(1))}
                    if mins_match.group(2):
                        result["minutes_high"] = int(mins_match.group(2))
                    return result

        return None

    except Exception as e:
        logger.debug(f"Error extracting time estimate: {e}")
        return None


async def extract_procedure_steps(page) -> Optional[List[str]]:
    """Extract repair procedure steps from page."""
    try:
        steps = []

        # Look for ordered lists or step containers
        selectors = [
            ".procedure-steps li",
            ".repair-steps li",
            "[class*='step'] li",
            "ol li",
            ".steps li",
        ]

        for selector in selectors:
            elements = await page.query_selector_all(selector)
            if elements and len(elements) >= 2:
                for elem in elements[:15]:  # Limit to 15 steps
                    text = await elem.inner_text()
                    text = sanitize_text(text, max_length=300)
                    if len(text) > 10:
                        steps.append(text)
                if steps:
                    break

        return steps if steps else None

    except Exception as e:
        logger.debug(f"Error extracting procedure steps: {e}")
        return None


async def extract_symptoms(page) -> Optional[List[str]]:
    """Extract related symptoms from page."""
    try:
        symptoms = []

        selectors = [
            ".symptoms li",
            "[class*='symptom'] li",
            ".warning-signs li",
        ]

        for selector in selectors:
            elements = await page.query_selector_all(selector)
            if elements:
                for elem in elements[:10]:
                    text = await elem.inner_text()
                    text = sanitize_text(text, max_length=200)
                    if len(text) > 5:
                        symptoms.append(text)
                if symptoms:
                    break

        return symptoms if symptoms else None

    except Exception as e:
        logger.debug(f"Error extracting symptoms: {e}")
        return None


async def extract_components(page) -> Optional[List[str]]:
    """Extract affected components from page."""
    try:
        components = []

        selectors = [
            ".components li",
            "[class*='part'] li",
            ".related-parts li",
        ]

        for selector in selectors:
            elements = await page.query_selector_all(selector)
            if elements:
                for elem in elements[:10]:
                    text = await elem.inner_text()
                    text = sanitize_text(text, max_length=100)
                    if len(text) > 3:
                        components.append(text)
                if components:
                    break

        return components if components else None

    except Exception as e:
        logger.debug(f"Error extracting components: {e}")
        return None


async def extract_vehicles(page) -> Optional[List[Dict[str, str]]]:
    """Extract affected vehicle list from page."""
    try:
        vehicles = []

        # Look for vehicle compatibility sections
        selectors = [
            ".vehicle-list li",
            "[class*='vehicle'] li",
            ".car-list li",
            ".makes-models li",
        ]

        for selector in selectors:
            elements = await page.query_selector_all(selector)
            if elements:
                for elem in elements[:20]:  # Limit to 20 vehicles
                    text = await elem.inner_text()
                    text = sanitize_text(text, max_length=100)

                    # Try to parse make/model/year
                    parts = text.split()
                    if len(parts) >= 2:
                        vehicles.append({"raw": text})

                if vehicles:
                    break

        return vehicles if vehicles else None

    except Exception as e:
        logger.debug(f"Error extracting vehicles: {e}")
        return None


def parse_price(price_str: str) -> Optional[float]:
    """Parse a price string to float."""
    try:
        # Remove $ and commas
        clean = price_str.replace('$', '').replace(',', '').strip()
        return float(clean)
    except (ValueError, AttributeError):
        return None


def get_related_dtc_codes(repair_type: str) -> List[str]:
    """Get DTC codes related to a repair type."""
    related = []

    for dtc, repairs in DTC_REPAIR_MAPPING.items():
        if repair_type in repairs:
            related.append(dtc)

    return related


async def scrape_all_repairs(
    repairs: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Scrape all repair types from RepairPal.

    Args:
        repairs: Optional list of repair types to scrape.
        limit: Optional limit on number of repairs to scrape.

    Returns:
        List of scraped repair data dictionaries.
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        logger.error("Playwright not installed. Run: pip install playwright && playwright install chromium")
        return []

    logger.info("Starting RepairPal scrape...")

    # First check robots.txt
    logger.info("Checking robots.txt...")
    robots_info = await check_robots_txt()

    if "error" in robots_info:
        logger.warning(f"Could not check robots.txt: {robots_info['error']}")
        logger.warning("Proceeding with caution using default rate limiting...")
    else:
        logger.info(f"robots.txt checked. Crawl delay: {robots_info.get('crawl_delay', 'not specified')}")

        # Respect crawl-delay if specified
        global RATE_LIMIT_DELAY
        if robots_info.get("crawl_delay"):
            RATE_LIMIT_DELAY = max(RATE_LIMIT_DELAY, robots_info["crawl_delay"])
            logger.info(f"Using crawl delay: {RATE_LIMIT_DELAY}s")

    repairs_to_scrape = repairs or COMMON_REPAIRS
    if limit:
        repairs_to_scrape = repairs_to_scrape[:limit]

    all_repairs = []

    async with async_playwright() as p:
        # Launch browser with stealth settings
        browser = await p.chromium.launch(
            headless=True,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--no-sandbox',
                '--disable-setuid-sandbox',
            ]
        )

        # Create context with realistic settings
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            locale='en-US',
        )

        # Add stealth script to avoid detection
        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
        """)

        page = await context.new_page()

        logger.info(f"Scraping {len(repairs_to_scrape)} repair types...")

        for i, repair_type in enumerate(repairs_to_scrape):
            logger.info(f"[{i+1}/{len(repairs_to_scrape)}] Scraping: {repair_type}")

            repair_data = await scrape_repair_page(page, repair_type)

            if repair_data:
                all_repairs.append(repair_data)
                logger.info(f"  Successfully scraped: {repair_data.get('name', repair_type)}")
            else:
                logger.debug(f"  No data found for: {repair_type}")

            # Rate limiting
            await asyncio.sleep(RATE_LIMIT_DELAY)

        await browser.close()

    logger.info(f"Scraped {len(all_repairs)} repairs from RepairPal")
    return all_repairs


def validate_repair_data(repair: Dict[str, Any]) -> bool:
    """Validate scraped repair data."""
    # Must have at least a name
    if not repair.get("name"):
        return False

    # Should have some useful data
    has_cost = repair.get("cost_low") or repair.get("cost_high") or repair.get("cost_average")
    has_description = repair.get("description") and len(repair.get("description", "")) > 20
    has_procedure = repair.get("procedure_steps") and len(repair.get("procedure_steps", [])) > 0

    return has_cost or has_description or has_procedure


def save_to_file(repairs: List[Dict[str, Any]], filepath: Path) -> None:
    """Save repairs to JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Filter and validate
    valid_repairs = [r for r in repairs if validate_repair_data(r)]

    data = {
        "metadata": {
            "source": "repairpal.com",
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "total_repairs": len(valid_repairs),
            "rate_limit_delay": RATE_LIMIT_DELAY,
        },
        "repairs": valid_repairs,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(valid_repairs)} repairs to {filepath}")


def load_from_file(filepath: Path) -> List[Dict[str, Any]]:
    """Load repairs from JSON file."""
    if not filepath.exists():
        return []

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get("repairs", [])


def map_repairs_to_dtc_codes(repairs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Create a mapping from DTC codes to relevant repairs.

    Args:
        repairs: List of repair data dictionaries.

    Returns:
        Dictionary mapping DTC codes to repair information.
    """
    dtc_repair_map = {}

    for repair in repairs:
        related_dtcs = repair.get("related_dtc_codes", [])

        for dtc in related_dtcs:
            if dtc not in dtc_repair_map:
                dtc_repair_map[dtc] = []

            dtc_repair_map[dtc].append({
                "repair_type": repair.get("repair_type"),
                "name": repair.get("name"),
                "cost_low": repair.get("cost_low"),
                "cost_high": repair.get("cost_high"),
                "labor_cost": repair.get("labor_cost"),
                "parts_cost": repair.get("parts_cost"),
                "time_estimate": repair.get("time_estimate"),
                "description": repair.get("description", "")[:200] if repair.get("description") else None,
            })

    return dtc_repair_map


def update_dtc_codes_with_repairs(repairs: List[Dict[str, Any]]) -> tuple[int, int]:
    """
    Update the DTC codes file with repair information.

    Args:
        repairs: List of repair data dictionaries.

    Returns:
        Tuple of (updated_count, total_dtc_count).
    """
    if not DTC_CODES_FILE.exists():
        logger.warning(f"DTC codes file not found: {DTC_CODES_FILE}")
        return 0, 0

    with open(DTC_CODES_FILE, "r", encoding="utf-8") as f:
        dtc_data = json.load(f)

    dtc_codes = {c["code"]: c for c in dtc_data.get("codes", [])}
    repair_map = map_repairs_to_dtc_codes(repairs)

    updated_count = 0

    for dtc_code, repair_info in repair_map.items():
        if dtc_code in dtc_codes:
            dtc_codes[dtc_code]["repair_estimates"] = repair_info
            updated_count += 1

    # Save updated DTC codes
    dtc_data["codes"] = list(dtc_codes.values())
    dtc_data["metadata"]["repair_data_added"] = datetime.now(timezone.utc).isoformat()
    dtc_data["metadata"]["repairs_mapped"] = updated_count

    with open(DTC_CODES_FILE, "w", encoding="utf-8") as f:
        json.dump(dtc_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Updated {updated_count} DTC codes with repair estimates")
    return updated_count, len(dtc_codes)


def generate_report(repairs: List[Dict[str, Any]]) -> str:
    """Generate a summary report of scraped repairs."""
    report = []
    report.append("=" * 70)
    report.append("REPAIRPAL SCRAPE SUMMARY")
    report.append("=" * 70)
    report.append(f"Total repairs scraped: {len(repairs)}")

    # Count repairs with cost data
    with_cost = sum(1 for r in repairs if r.get("cost_low") or r.get("cost_high"))
    report.append(f"Repairs with cost estimates: {with_cost}")

    # Count repairs with procedure steps
    with_procedure = sum(1 for r in repairs if r.get("procedure_steps"))
    report.append(f"Repairs with procedure steps: {with_procedure}")

    # Count repairs with time estimates
    with_time = sum(1 for r in repairs if r.get("time_estimate"))
    report.append(f"Repairs with time estimates: {with_time}")

    # Count repairs mapped to DTC codes
    with_dtc = sum(1 for r in repairs if r.get("related_dtc_codes"))
    report.append(f"Repairs mapped to DTC codes: {with_dtc}")

    # List repairs with cost data
    report.append("\n" + "-" * 70)
    report.append("REPAIRS WITH COST ESTIMATES:")
    report.append("-" * 70)

    for repair in sorted(repairs, key=lambda x: x.get("name", "")):
        if repair.get("cost_low") or repair.get("cost_high"):
            name = repair.get("name", repair.get("repair_type", "Unknown"))
            cost_low = repair.get("cost_low", "N/A")
            cost_high = repair.get("cost_high", "N/A")
            dtcs = repair.get("related_dtc_codes", [])

            cost_str = f"${cost_low}" if cost_low != "N/A" else ""
            if cost_high != "N/A":
                cost_str += f" - ${cost_high}" if cost_str else f"${cost_high}"

            dtc_str = f" (DTCs: {', '.join(dtcs[:3])}{'...' if len(dtcs) > 3 else ''})" if dtcs else ""
            report.append(f"  {name}: {cost_str}{dtc_str}")

    report.append("=" * 70)

    return "\n".join(report)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Scrape repair data from RepairPal")
    parser.add_argument("--scrape", action="store_true", help="Scrape repairs from website")
    parser.add_argument("--merge", action="store_true", help="Merge with DTC codes file")
    parser.add_argument("--use-cache", action="store_true", help="Use cached data")
    parser.add_argument("--check-robots", action="store_true", help="Only check robots.txt")
    parser.add_argument("--limit", type=int, help="Limit number of repairs to scrape")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # If only checking robots.txt
    if args.check_robots:
        robots_info = await check_robots_txt()
        print(json.dumps(robots_info, indent=2))
        return

    # Default behavior: scrape and merge
    if not (args.scrape or args.merge or args.use_cache):
        args.scrape = True
        args.merge = True

    try:
        if args.use_cache:
            repairs = load_from_file(CACHE_FILE)
            logger.info(f"Loaded {len(repairs)} repairs from cache")
        elif args.scrape:
            repairs = await scrape_all_repairs(limit=args.limit)
            save_to_file(repairs, CACHE_FILE)
        else:
            repairs = load_from_file(CACHE_FILE)

        if not repairs:
            logger.warning("No repair data available.")
            return

        # Generate and print report
        report = generate_report(repairs)
        print(report)

        # Merge with DTC codes if requested
        if args.merge:
            updated, total = update_dtc_codes_with_repairs(repairs)
            print(f"\nMerge results: {updated} DTC codes updated out of {total} total")

    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
