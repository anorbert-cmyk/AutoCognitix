#!/usr/bin/env python3
"""
iFixit Automotive Repair Guide Scraper

Scrapes repair guides from iFixit's Car and Truck section with:
- Async scraping with aiohttp + BeautifulSoup
- robots.txt compliance (no /api/*, /edit/*, search pages)
- Rate limiting (1 request/second default)
- Checkpoint/resume support
- Progress tracking with tqdm
- Comprehensive JSON output

iFixit is CC BY-NC-SA 3.0 licensed - for non-commercial research only.

Usage:
    python scripts/scrape_ifixit_automotive.py              # Scrape all
    python scripts/scrape_ifixit_automotive.py --list       # List categories only
    python scripts/scrape_ifixit_automotive.py --resume     # Resume from checkpoint
    python scripts/scrape_ifixit_automotive.py --stats      # Show statistics
    python scripts/scrape_ifixit_automotive.py --limit 10   # Limit guides per category
"""

import argparse
import asyncio
import json
import logging
import re
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin, urlparse, unquote

import aiohttp
from bs4 import BeautifulSoup, Tag
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "ifixit"
GUIDES_DIR = DATA_DIR / "guides"
CATEGORIES_DIR = DATA_DIR / "categories"
CHECKPOINT_DIR = PROJECT_ROOT / "scripts" / "checkpoints"
CHECKPOINT_FILE = CHECKPOINT_DIR / "ifixit_scrape_state.json"

# iFixit configuration
BASE_URL = "https://www.ifixit.com"
CAR_TRUCK_URL = f"{BASE_URL}/Device/Car_and_Truck"
GUIDE_IMAGE_CDN = "https://guide-images.cdn.ifixit.com"

# Scraping settings
DEFAULT_RATE_LIMIT = 1.0  # Seconds between requests (1 req/sec)
TIMEOUT = 30.0  # HTTP timeout
MAX_RETRIES = 3
RETRY_DELAY = 5.0  # Seconds to wait before retry
USER_AGENT = (
    "AutoCognitix-Research-Bot/1.0 "
    "(https://github.com/autocognitix; non-commercial research; "
    "respects robots.txt; contact: research@autocognitix.com)"
)

# Paths that are disallowed per robots.txt
DISALLOWED_PATTERNS = [
    "/api/",
    "/edit/",
    "/History/",
    "/Search",
    "/Translate/",
    "/Moderation/",
]


@dataclass
class ScrapeState:
    """Tracks scraping progress for resume capability."""

    scraped_categories: set[str] = field(default_factory=set)
    scraped_guides: set[str] = field(default_factory=set)
    failed_urls: set[str] = field(default_factory=set)
    total_guides: int = 0
    last_updated: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "scraped_categories": list(self.scraped_categories),
            "scraped_guides": list(self.scraped_guides),
            "failed_urls": list(self.failed_urls),
            "total_guides": self.total_guides,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScrapeState":
        return cls(
            scraped_categories=set(data.get("scraped_categories", [])),
            scraped_guides=set(data.get("scraped_guides", [])),
            failed_urls=set(data.get("failed_urls", [])),
            total_guides=data.get("total_guides", 0),
            last_updated=data.get("last_updated", ""),
        )


@dataclass
class RepairGuide:
    """Represents an iFixit repair guide."""

    guide_id: int
    url: str
    title: str
    device: str
    category: str
    difficulty: Optional[str] = None
    time_required: Optional[str] = None
    introduction: Optional[str] = None
    conclusion: Optional[str] = None
    tools: list[dict[str, str]] = field(default_factory=list)
    parts: list[dict[str, str]] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    author: Optional[str] = None
    contributors: list[str] = field(default_factory=list)
    views: Optional[int] = None
    flags: list[str] = field(default_factory=list)
    scraped_at: str = ""
    license: str = "CC BY-NC-SA 3.0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Category:
    """Represents an iFixit device category."""

    name: str
    url: str
    parent: Optional[str] = None
    subcategories: list[str] = field(default_factory=list)
    guide_count: int = 0
    guide_urls: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class IFixitScraper:
    """Async scraper for iFixit automotive repair guides."""

    def __init__(
        self,
        rate_limit: float = DEFAULT_RATE_LIMIT,
        resume: bool = False,
        limit_per_category: Optional[int] = None,
    ):
        self.rate_limit = rate_limit
        self.limit_per_category = limit_per_category
        self.state = ScrapeState()
        self.session: Optional[aiohttp.ClientSession] = None
        self._last_request_time = 0.0
        self._semaphore = asyncio.Semaphore(1)  # Single request at a time for rate limiting

        # Create directories
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        GUIDES_DIR.mkdir(parents=True, exist_ok=True)
        CATEGORIES_DIR.mkdir(parents=True, exist_ok=True)
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

        # Load checkpoint if resuming
        if resume and CHECKPOINT_FILE.exists():
            self._load_checkpoint()
            logger.info(
                f"Resumed from checkpoint: {len(self.state.scraped_guides)} guides, "
                f"{len(self.state.scraped_categories)} categories"
            )

    def _load_checkpoint(self) -> None:
        """Load scraping state from checkpoint file."""
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.state = ScrapeState.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")

    def _save_checkpoint(self) -> None:
        """Save current scraping state to checkpoint file."""
        self.state.last_updated = datetime.now(timezone.utc).isoformat()
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump(self.state.to_dict(), f, indent=2)

    def _is_allowed_url(self, url: str) -> bool:
        """Check if URL is allowed per robots.txt rules."""
        parsed = urlparse(url)
        path = parsed.path

        for pattern in DISALLOWED_PATTERNS:
            if pattern in path:
                return False
        return True

    async def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        async with self._semaphore:
            now = asyncio.get_event_loop().time()
            elapsed = now - self._last_request_time
            if elapsed < self.rate_limit:
                await asyncio.sleep(self.rate_limit - elapsed)
            self._last_request_time = asyncio.get_event_loop().time()

    async def _fetch(self, url: str, retries: int = MAX_RETRIES) -> Optional[str]:
        """Fetch a URL with rate limiting and retries."""
        if not self._is_allowed_url(url):
            logger.warning(f"URL blocked by robots.txt rules: {url}")
            return None

        await self._rate_limit()

        for attempt in range(retries):
            try:
                async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as response:
                    if response.status == 200:
                        return await response.text()
                    elif response.status == 404:
                        logger.warning(f"Not found (404): {url}")
                        return None
                    elif response.status == 429:
                        # Rate limited - wait and retry
                        wait_time = RETRY_DELAY * (attempt + 1)
                        logger.warning(f"Rate limited (429), waiting {wait_time}s: {url}")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.warning(f"HTTP {response.status}: {url}")
                        if attempt < retries - 1:
                            await asyncio.sleep(RETRY_DELAY)
            except asyncio.TimeoutError:
                logger.warning(f"Timeout (attempt {attempt + 1}/{retries}): {url}")
                if attempt < retries - 1:
                    await asyncio.sleep(RETRY_DELAY)
            except aiohttp.ClientError as e:
                logger.warning(f"Client error (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(RETRY_DELAY)

        self.state.failed_urls.add(url)
        return None

    async def _get_soup(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch URL and parse with BeautifulSoup."""
        html = await self._fetch(url)
        if html:
            return BeautifulSoup(html, "html.parser")
        return None

    def _extract_guide_id(self, url: str) -> Optional[int]:
        """Extract guide ID from URL."""
        # Pattern: /Guide/Title/12345
        match = re.search(r"/Guide/[^/]+/(\d+)", url)
        if match:
            return int(match.group(1))
        return None

    def _clean_text(self, text: Optional[str]) -> Optional[str]:
        """Clean and normalize text content."""
        if not text:
            return None
        # Remove excessive whitespace while preserving paragraph structure
        text = re.sub(r"\n\s*\n", "\n\n", text.strip())
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    async def get_categories(self) -> list[Category]:
        """Get all automotive categories from the main page."""
        logger.info(f"Fetching categories from {CAR_TRUCK_URL}")

        soup = await self._get_soup(CAR_TRUCK_URL)
        if not soup:
            logger.error("Failed to fetch category page")
            return []

        categories = []

        # Find category links - they're typically in .blurbListWide or device grid
        # Look for links to /Device/ pages
        device_links = soup.find_all("a", href=re.compile(r"^/Device/[^/]+$"))

        seen_urls = set()
        for link in device_links:
            href = link.get("href", "")
            if not href or href in seen_urls:
                continue

            # Skip the parent category link
            if href == "/Device/Car_and_Truck":
                continue

            seen_urls.add(href)
            full_url = urljoin(BASE_URL, href)

            # Extract category name from URL or link text
            name = link.get_text(strip=True)
            if not name:
                # Try to get name from URL
                name = unquote(href.split("/")[-1].replace("_", " "))

            categories.append(Category(
                name=name,
                url=full_url,
                parent="Car_and_Truck",
            ))

        logger.info(f"Found {len(categories)} categories")
        return categories

    def _is_valid_device_url(self, href: str) -> bool:
        """Check if a device URL is valid (not Edit, History, etc.)."""
        if not href:
            return False

        # Must be a Device URL
        if not href.startswith("/Device/"):
            return False

        # Extract the device name part
        parts = href.split("/")
        if len(parts) < 3:
            return False

        device_name = parts[2]

        # Skip special pages (Edit, History, etc.)
        invalid_names = {"Edit", "History", "Translate", "Search", "Moderation"}
        if device_name in invalid_names:
            return False

        return True

    async def get_subcategories(self, category: Category) -> list[Category]:
        """Get subcategories (e.g., year ranges) from a category page."""
        soup = await self._get_soup(category.url)
        if not soup:
            return []

        subcategories = []

        # Find links to child device pages
        device_links = soup.find_all("a", href=re.compile(r"^/Device/"))

        seen_urls = set()
        for link in device_links:
            href = link.get("href", "")

            # Validate the URL
            if not self._is_valid_device_url(href):
                continue

            full_url = urljoin(BASE_URL, href)

            # Skip self-references and already seen
            if full_url == category.url or full_url in seen_urls:
                continue

            # Skip if it goes back up the hierarchy
            if href == "/Device/Car_and_Truck":
                continue

            seen_urls.add(full_url)

            name = link.get_text(strip=True)
            if not name:
                name = unquote(href.split("/")[-1].replace("_", " "))

            subcategories.append(Category(
                name=name,
                url=full_url,
                parent=category.name,
            ))

        category.subcategories = [sc.name for sc in subcategories]
        return subcategories

    async def get_guide_urls(self, category: Category) -> list[str]:
        """Extract guide URLs from a category page."""
        soup = await self._get_soup(category.url)
        if not soup:
            return []

        guide_urls = []

        # Find all guide links - pattern: /Guide/*/number
        guide_links = soup.find_all("a", href=re.compile(r"^/Guide/[^/]+/\d+"))

        seen_ids = set()
        for link in guide_links:
            href = link.get("href", "")
            guide_id = self._extract_guide_id(href)

            if guide_id and guide_id not in seen_ids:
                seen_ids.add(guide_id)
                full_url = urljoin(BASE_URL, href)
                guide_urls.append(full_url)

        category.guide_count = len(guide_urls)
        category.guide_urls = guide_urls

        return guide_urls

    def _parse_difficulty(self, soup: BeautifulSoup) -> Optional[str]:
        """Parse difficulty level from guide page."""
        # Look for difficulty indicator
        difficulty_elem = soup.find(string=re.compile(r"Difficulty", re.I))
        if difficulty_elem:
            parent = difficulty_elem.find_parent()
            if parent:
                # Get the sibling or child that contains the actual level
                text = parent.get_text(strip=True)
                # Extract difficulty level (Easy, Moderate, Difficult, Very Difficult)
                match = re.search(r"(Very\s+)?(Easy|Moderate|Difficult|Hard)", text, re.I)
                if match:
                    return match.group(0)

        # Try meta tags
        meta_diff = soup.find("meta", {"name": "difficulty"})
        if meta_diff and meta_diff.get("content"):
            return meta_diff["content"]

        return None

    def _parse_time(self, soup: BeautifulSoup) -> Optional[str]:
        """Parse time required from guide page."""
        # Look for time indicator
        time_patterns = [
            r"(\d+(?:\s*-\s*\d+)?)\s*(minutes?|hours?|mins?|hrs?)",
            r"Time Required[:\s]*([^<\n]+)",
        ]

        text = soup.get_text()
        for pattern in time_patterns:
            match = re.search(pattern, text, re.I)
            if match:
                return match.group(0).strip()

        return None

    def _parse_tools(self, soup: BeautifulSoup) -> list[dict[str, str]]:
        """Parse tools needed from guide page."""
        tools = []

        # Look for tools/requirements section
        # Common patterns: "What you need", "Tools", "Required"
        tools_section = None

        for heading in soup.find_all(["h2", "h3", "h4", "div", "span"]):
            text = heading.get_text(strip=True).lower()
            if "what you need" in text or "tools" in text or "required" in text:
                tools_section = heading
                break

        if tools_section:
            # Look for list items after the heading
            parent = tools_section.find_parent(["div", "section"])
            if parent:
                for item in parent.find_all(["li", "a"]):
                    if item.name == "a" and "/Item/" in item.get("href", ""):
                        tools.append({
                            "name": item.get_text(strip=True),
                            "url": urljoin(BASE_URL, item.get("href", "")),
                        })
                    elif item.name == "li":
                        text = item.get_text(strip=True)
                        if text and len(text) < 200:  # Reasonable tool name length
                            link = item.find("a")
                            tools.append({
                                "name": text,
                                "url": urljoin(BASE_URL, link.get("href", "")) if link else "",
                            })

        return tools

    def _parse_parts(self, soup: BeautifulSoup) -> list[dict[str, str]]:
        """Parse parts needed from guide page."""
        parts = []

        # Look for parts section
        parts_section = None

        for heading in soup.find_all(["h2", "h3", "h4", "div", "span"]):
            text = heading.get_text(strip=True).lower()
            if "parts" in text or "materials" in text:
                parts_section = heading
                break

        if parts_section:
            parent = parts_section.find_parent(["div", "section"])
            if parent:
                for item in parent.find_all("li"):
                    text = item.get_text(strip=True)
                    if text and len(text) < 200:
                        link = item.find("a")
                        parts.append({
                            "name": text,
                            "url": urljoin(BASE_URL, link.get("href", "")) if link else "",
                        })

        return parts

    def _parse_steps(self, soup: BeautifulSoup) -> list[dict[str, Any]]:
        """Parse step-by-step procedures from guide page."""
        steps = []

        # Find step headings - typically h2 or h3 with "Step N"
        step_headings = soup.find_all(["h2", "h3"], string=re.compile(r"Step\s+\d+", re.I))

        if not step_headings:
            # Try finding steps by ID pattern
            step_elements = soup.find_all(id=re.compile(r"^s\d+"))
            for elem in step_elements:
                step_num = len(steps) + 1
                step_data = self._extract_step_data(elem, step_num)
                if step_data:
                    steps.append(step_data)
            return steps

        for heading in step_headings:
            # Extract step number
            match = re.search(r"Step\s+(\d+)", heading.get_text(), re.I)
            if not match:
                continue

            step_num = int(match.group(1))
            step_data = self._extract_step_data(heading, step_num)
            if step_data:
                steps.append(step_data)

        return steps

    def _clean_ui_noise(self, text: str) -> str:
        """Remove iFixit UI elements from text."""
        # Remove common UI patterns
        ui_patterns = [
            r"Edit\s*$",
            r"Add a comment.*$",
            r"Add Comment.*$",
            r"\d+\s*Cancel\s*Post\s*comment",
            r"Post\s*comment\s*$",
            r"Cancel\s*$",
            r"^\s*\d+\s*$",  # Standalone numbers (comment counts)
        ]
        for pattern in ui_patterns:
            text = re.sub(pattern, "", text, flags=re.I)
        return text.strip()

    def _is_valid_instruction(self, text: str) -> bool:
        """Check if text is a valid instruction (not UI noise)."""
        if not text or len(text) < 5:
            return False

        # Filter out pure UI elements
        ui_texts = {
            "edit", "add a comment", "cancel", "post comment",
            "add comment", "cancel post", "follow this guide"
        }
        if text.lower().strip() in ui_texts:
            return False

        # Filter out step navigation
        if re.match(r"^(step\s+\d+|next|previous|back)$", text, re.I):
            return False

        return True

    def _extract_step_data(self, element: Tag, step_num: int) -> Optional[dict[str, Any]]:
        """Extract data from a step element."""
        step_data = {
            "step_number": step_num,
            "title": "",
            "instructions": [],
            "warnings": [],
            "notes": [],
            "images": [],
        }

        # Get step container (parent div/section)
        container = element
        if element.name in ["h2", "h3"]:
            container = element.find_parent(["div", "section", "li"])
            if not container:
                container = element.parent

        if not container:
            return None

        # Get step title (text after "Step N") - clean it
        heading_text = element.get_text(strip=True)
        title_text = re.sub(r"^Step\s+\d+\s*", "", heading_text, flags=re.I)
        title_text = self._clean_ui_noise(title_text)
        if title_text:
            step_data["title"] = title_text

        # Track seen instructions to avoid duplicates
        seen_instructions = set()

        # Get instructions (bullet points, paragraphs)
        for item in container.find_all(["li", "p"]):
            text = item.get_text(strip=True)
            text = self._clean_ui_noise(text)

            if not self._is_valid_instruction(text):
                continue

            # Normalize for duplicate detection
            normalized = text.lower().strip()
            if normalized in seen_instructions:
                continue
            seen_instructions.add(normalized)

            # Check for warnings/cautions
            if re.match(r"^(warning|caution|danger):", text, re.I):
                step_data["warnings"].append(text)
            elif re.match(r"^(note|tip|hint):", text, re.I):
                step_data["notes"].append(text)
            else:
                step_data["instructions"].append(text)

        # Get images - deduplicate
        seen_images = set()
        for img in container.find_all("img"):
            src = img.get("src", "") or img.get("data-src", "")
            if src and "guide-images" in src:
                # Get largest version by modifying URL
                if ".thumbnail" in src:
                    src = src.replace(".thumbnail", ".standard")
                elif ".200x150" in src:
                    src = re.sub(r"\.\d+x\d+", ".standard", src)

                # Deduplicate by base image ID
                base_id = re.search(r"/igi/([^.]+)", src)
                if base_id:
                    img_key = base_id.group(1)
                    if img_key not in seen_images:
                        seen_images.add(img_key)
                        step_data["images"].append(src)

        # Only return if we have content
        if step_data["instructions"] or step_data["images"]:
            return step_data
        return None

    def _parse_images(self, soup: BeautifulSoup) -> list[str]:
        """Extract all guide image URLs."""
        images = []
        seen = set()

        for img in soup.find_all("img"):
            src = img.get("src", "") or img.get("data-src", "")
            if src and "guide-images" in src and src not in seen:
                seen.add(src)
                # Normalize to standard size
                if ".thumbnail" in src:
                    src = src.replace(".thumbnail", ".medium")
                images.append(src)

        return images

    def _parse_introduction(self, soup: BeautifulSoup) -> Optional[str]:
        """Parse guide introduction text."""
        # Look for introduction section
        intro_heading = soup.find(["h2", "h3"], string=re.compile(r"Introduction", re.I))
        if intro_heading:
            next_elem = intro_heading.find_next_sibling()
            if next_elem:
                return self._clean_text(next_elem.get_text())

        # Try meta description
        meta_desc = soup.find("meta", {"name": "description"})
        if meta_desc and meta_desc.get("content"):
            return meta_desc["content"]

        return None

    def _parse_conclusion(self, soup: BeautifulSoup) -> Optional[str]:
        """Parse guide conclusion text."""
        conclusion_heading = soup.find(["h2", "h3"], string=re.compile(r"Conclusion", re.I))
        if conclusion_heading:
            next_elem = conclusion_heading.find_next_sibling()
            if next_elem:
                return self._clean_text(next_elem.get_text())
        return None

    async def scrape_guide(self, url: str, category: str, device: str) -> Optional[RepairGuide]:
        """Scrape a single repair guide."""
        guide_id = self._extract_guide_id(url)
        if not guide_id:
            logger.warning(f"Could not extract guide ID from: {url}")
            return None

        # Check if already scraped
        guide_file = GUIDES_DIR / f"guide_{guide_id}.json"
        if str(guide_id) in self.state.scraped_guides and guide_file.exists():
            logger.debug(f"Skipping already scraped guide: {guide_id}")
            return None

        soup = await self._get_soup(url)
        if not soup:
            return None

        # Extract title
        title_elem = soup.find(["h1", "h2"], class_=re.compile(r"title", re.I))
        if not title_elem:
            title_elem = soup.find("h1")
        title = title_elem.get_text(strip=True) if title_elem else f"Guide {guide_id}"

        # Build guide object
        guide = RepairGuide(
            guide_id=guide_id,
            url=url,
            title=title,
            device=device,
            category=category,
            difficulty=self._parse_difficulty(soup),
            time_required=self._parse_time(soup),
            introduction=self._parse_introduction(soup),
            conclusion=self._parse_conclusion(soup),
            tools=self._parse_tools(soup),
            parts=self._parse_parts(soup),
            steps=self._parse_steps(soup),
            images=self._parse_images(soup),
            scraped_at=datetime.now(timezone.utc).isoformat(),
        )

        # Extract author if available
        author_elem = soup.find(["a", "span"], class_=re.compile(r"author|creator", re.I))
        if author_elem:
            guide.author = author_elem.get_text(strip=True)

        # Save guide
        with open(guide_file, "w", encoding="utf-8") as f:
            json.dump(guide.to_dict(), f, indent=2, ensure_ascii=False)

        self.state.scraped_guides.add(str(guide_id))
        self.state.total_guides += 1

        return guide

    async def scrape_category(self, category: Category) -> list[RepairGuide]:
        """Scrape all guides from a category and its subcategories."""
        guides = []

        # Check if already scraped
        if category.url in self.state.scraped_categories:
            logger.debug(f"Skipping already scraped category: {category.name}")
            return guides

        logger.info(f"Scraping category: {category.name}")

        # Get subcategories first
        subcategories = await self.get_subcategories(category)

        # Get guides from this category
        guide_urls = await self.get_guide_urls(category)

        # Apply limit if set
        if self.limit_per_category and len(guide_urls) > self.limit_per_category:
            guide_urls = guide_urls[:self.limit_per_category]

        # Scrape guides
        for url in tqdm(guide_urls, desc=f"  {category.name}", leave=False):
            guide = await self.scrape_guide(url, category=category.name, device=category.name)
            if guide:
                guides.append(guide)

            # Save checkpoint periodically
            if len(guides) % 10 == 0:
                self._save_checkpoint()

        # Save category
        category_file = CATEGORIES_DIR / f"{category.name.replace(' ', '_').replace('/', '_')}.json"
        with open(category_file, "w", encoding="utf-8") as f:
            json.dump(category.to_dict(), f, indent=2, ensure_ascii=False)

        self.state.scraped_categories.add(category.url)

        # Recursively scrape subcategories
        for subcat in subcategories:
            sub_guides = await self.scrape_category(subcat)
            guides.extend(sub_guides)

        return guides

    async def run(self) -> dict[str, Any]:
        """Run the full scraping process."""
        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

        async with aiohttp.ClientSession(headers=headers) as session:
            self.session = session

            # Get all categories
            categories = await self.get_categories()

            all_guides = []

            # Scrape each category
            for category in tqdm(categories, desc="Categories"):
                guides = await self.scrape_category(category)
                all_guides.extend(guides)
                self._save_checkpoint()

            # Generate summary
            summary = {
                "total_categories": len(categories),
                "total_guides": len(all_guides),
                "scraped_at": datetime.now(timezone.utc).isoformat(),
                "categories": [c.to_dict() for c in categories],
                "failed_urls": list(self.state.failed_urls),
            }

            # Save summary
            summary_file = DATA_DIR / "scrape_summary.json"
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            logger.info(f"Scraping complete: {len(all_guides)} guides from {len(categories)} categories")

            return summary

    async def list_categories(self) -> None:
        """List all categories without scraping."""
        headers = {"User-Agent": USER_AGENT}

        async with aiohttp.ClientSession(headers=headers) as session:
            self.session = session

            categories = await self.get_categories()

            print(f"\n{'='*60}")
            print(f"iFixit Automotive Categories ({len(categories)} total)")
            print(f"{'='*60}\n")

            for cat in sorted(categories, key=lambda c: c.name):
                print(f"  - {cat.name}")
                print(f"    URL: {cat.url}")

                # Get subcategories
                subcats = await self.get_subcategories(cat)
                if subcats:
                    print(f"    Subcategories: {len(subcats)}")
                    for sc in subcats[:5]:
                        print(f"      - {sc.name}")
                    if len(subcats) > 5:
                        print(f"      ... and {len(subcats) - 5} more")

                # Get guide count
                guide_urls = await self.get_guide_urls(cat)
                print(f"    Guides: {len(guide_urls)}")
                print()


def show_stats() -> None:
    """Show scraping statistics."""
    if not CHECKPOINT_FILE.exists():
        print("No checkpoint file found. Run the scraper first.")
        return

    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        state = json.load(f)

    print(f"\n{'='*60}")
    print("iFixit Scraper Statistics")
    print(f"{'='*60}\n")
    print(f"Categories scraped: {len(state.get('scraped_categories', []))}")
    print(f"Guides scraped: {len(state.get('scraped_guides', []))}")
    print(f"Failed URLs: {len(state.get('failed_urls', []))}")
    print(f"Last updated: {state.get('last_updated', 'N/A')}")

    # Count files
    if GUIDES_DIR.exists():
        guide_files = list(GUIDES_DIR.glob("guide_*.json"))
        print(f"Guide files on disk: {len(guide_files)}")

    if CATEGORIES_DIR.exists():
        cat_files = list(CATEGORIES_DIR.glob("*.json"))
        print(f"Category files on disk: {len(cat_files)}")

    # Show failed URLs if any
    failed = state.get("failed_urls", [])
    if failed:
        print(f"\nFailed URLs ({len(failed)}):")
        for url in failed[:10]:
            print(f"  - {url}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")


async def main():
    parser = argparse.ArgumentParser(
        description="Scrape iFixit automotive repair guides",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/scrape_ifixit_automotive.py              # Full scrape
    python scripts/scrape_ifixit_automotive.py --list       # List categories
    python scripts/scrape_ifixit_automotive.py --resume     # Resume from checkpoint
    python scripts/scrape_ifixit_automotive.py --limit 5    # 5 guides per category
    python scripts/scrape_ifixit_automotive.py --rate 2.0   # 2 seconds between requests

Note: iFixit content is CC BY-NC-SA 3.0 licensed - for non-commercial use only.
        """,
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all categories without scraping",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show scraping statistics",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit guides per category (for testing)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=DEFAULT_RATE_LIMIT,
        help=f"Rate limit in seconds (default: {DEFAULT_RATE_LIMIT})",
    )

    args = parser.parse_args()

    if args.stats:
        show_stats()
        return

    scraper = IFixitScraper(
        rate_limit=args.rate,
        resume=args.resume,
        limit_per_category=args.limit,
    )

    if args.list:
        await scraper.list_categories()
    else:
        summary = await scraper.run()
        print(f"\nScraping complete!")
        print(f"Total guides: {summary['total_guides']}")
        print(f"Total categories: {summary['total_categories']}")
        print(f"Failed URLs: {len(summary['failed_urls'])}")
        print(f"\nData saved to: {DATA_DIR}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Progress saved to checkpoint.")
        sys.exit(1)
