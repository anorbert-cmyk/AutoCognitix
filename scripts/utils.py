#!/usr/bin/env python3
"""
Shared utilities for AutoCognitix import scripts.

This module provides common functionality used across multiple scripts:
- Database URL conversion
- Input sanitization
- Logging configuration
- Path validation
"""

import html
import logging
import re
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Project root path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Configure module logger
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging for scripts.

    Args:
        verbose: If True, set DEBUG level, otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_sync_db_url() -> str:
    """
    Convert async database URL to sync for direct database operations.

    Returns:
        Synchronous PostgreSQL connection URL.

    Raises:
        ValueError: If DATABASE_URL is not configured.
    """
    # Import settings lazily to avoid circular imports
    sys.path.insert(0, str(PROJECT_ROOT))
    from backend.app.core.config import settings

    url = settings.DATABASE_URL

    if not url:
        raise ValueError("DATABASE_URL not configured")

    # Convert async driver to sync
    if url.startswith("postgresql+asyncpg://"):
        url = url.replace("postgresql+asyncpg://", "postgresql://")

    return url


def get_utc_now() -> datetime:
    """
    Get current UTC datetime (timezone-aware).

    Returns:
        Current datetime in UTC timezone.
    """
    return datetime.now(timezone.utc)


def sanitize_text(
    text: str,
    max_length: int = 2000,
    remove_html: bool = True,
    remove_control_chars: bool = True,
) -> str:
    """
    Sanitize text input from external sources.

    Args:
        text: Input text to sanitize.
        max_length: Maximum allowed length.
        remove_html: Whether to strip HTML tags.
        remove_control_chars: Whether to remove control characters.

    Returns:
        Sanitized text string.
    """
    if not text:
        return ""

    # Strip whitespace
    text = text.strip()

    # Remove HTML tags if requested
    if remove_html:
        text = re.sub(r'<[^>]+>', '', text)
        text = html.unescape(text)

    # Remove control characters if requested
    if remove_control_chars:
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # Truncate to max length
    if len(text) > max_length:
        text = text[:max_length]

    return text


def validate_dtc_code(code: str) -> bool:
    """
    Validate DTC code format.

    Args:
        code: DTC code string to validate.

    Returns:
        True if valid DTC code format, False otherwise.
    """
    if not code:
        return False

    # Standard OBD-II DTC format: [PCBU][0-9A-F]{4}
    pattern = r'^[PCBU][0-9A-F]{4}$'
    return bool(re.match(pattern, code.upper()))


def get_category_from_code(code: str) -> str:
    """
    Determine the category from a DTC code prefix.

    Args:
        code: DTC code string.

    Returns:
        Category string (powertrain, chassis, body, network, or unknown).
    """
    if not code:
        return "unknown"

    prefix = code[0].upper()
    categories = {
        "P": "powertrain",
        "C": "chassis",
        "B": "body",
        "U": "network",
    }
    return categories.get(prefix, "unknown")


def get_severity_from_code(code: str) -> str:
    """
    Estimate severity based on DTC code pattern.

    Args:
        code: DTC code string.

    Returns:
        Severity level (low, medium, high, critical).
    """
    if not code:
        return "medium"

    prefix = code[0].upper()
    code_upper = code.upper()

    # Network codes are typically high severity
    if prefix == "U":
        return "high"

    # Body codes for safety systems
    if prefix == "B" and code_upper.startswith(("B0", "B1")):
        return "critical"

    # Powertrain codes
    if prefix == "P":
        if code_upper.startswith("P03"):  # Misfire
            return "high"
        if code_upper.startswith(("P07", "P08", "P09")):  # Transmission
            return "high"

    return "medium"


def get_system_from_code(code: str) -> str:
    """
    Determine the system from a DTC code.

    Args:
        code: DTC code string.

    Returns:
        System description string.
    """
    if not code or len(code) < 3:
        return ""

    prefix = code[0].upper()
    middle = code[1:3]

    if prefix == "P":
        systems = {
            "00": "Fuel and Air Metering",
            "01": "Fuel and Air Metering",
            "02": "Fuel and Air Metering Injection",
            "03": "Ignition System/Misfire",
            "04": "Auxiliary Emission Controls",
            "05": "Vehicle Speed and Idle Control",
            "06": "Computer Output Circuits",
            "07": "Transmission",
            "08": "Transmission",
            "09": "Transmission",
            "0A": "Hybrid Propulsion",
        }
        return systems.get(middle, "Powertrain")
    elif prefix == "C":
        return "Chassis/ABS/Traction"
    elif prefix == "B":
        return "Body/Interior"
    elif prefix == "U":
        return "Network Communication"

    return ""


def validate_path_within_project(path: Path) -> bool:
    """
    Validate that a path is within the project root.

    Args:
        path: Path to validate.

    Returns:
        True if path is within project root, False otherwise.
    """
    try:
        resolved = path.resolve()
        return str(resolved).startswith(str(PROJECT_ROOT))
    except Exception:
        return False


def safe_json_loads(
    content: str,
    max_size: int = 10 * 1024 * 1024,  # 10MB
    max_depth: int = 50,
) -> Optional[Dict[str, Any]]:
    """
    Safely parse JSON with size and depth limits.

    Args:
        content: JSON string to parse.
        max_size: Maximum allowed string size in bytes.
        max_depth: Maximum nesting depth.

    Returns:
        Parsed JSON dict or None if parsing fails.
    """
    import json

    if not content or len(content) > max_size:
        logger.warning(f"JSON content too large: {len(content) if content else 0} bytes")
        return None

    try:
        # Check approximate nesting depth
        depth = content.count('{') + content.count('[')
        if depth > max_depth * 2:  # Rough estimate
            logger.warning(f"JSON appears too deeply nested: {depth} brackets")
            return None

        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON decode error: {e}")
        return None


def normalize_dtc_codes(codes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize and deduplicate DTC codes.

    Args:
        codes: List of DTC code dictionaries.

    Returns:
        Normalized, deduplicated list of codes.
    """
    seen = set()
    normalized = []

    for code_data in codes:
        code = code_data.get("code", "").upper().strip()

        if not code or code in seen:
            continue

        if not validate_dtc_code(code):
            continue

        seen.add(code)

        description = code_data.get("description_en", "")
        if description:
            description = sanitize_text(description)

        if not description:
            continue

        normalized.append({
            "code": code,
            "description_en": description,
            "description_hu": code_data.get("description_hu"),
            "category": get_category_from_code(code),
            "severity": get_severity_from_code(code),
            "system": get_system_from_code(code),
            "is_generic": code[1] == "0",  # Boolean, not string!
            "symptoms": code_data.get("symptoms", []),
            "possible_causes": code_data.get("possible_causes", []),
            "diagnostic_steps": code_data.get("diagnostic_steps", []),
            "related_codes": code_data.get("related_codes", []),
            "source": code_data.get("source", "unknown"),
            "manufacturer": code_data.get("manufacturer"),
            "translation_status": code_data.get("translation_status", "pending"),
        })

    normalized.sort(key=lambda x: x["code"])
    return normalized


class DatabaseImporter:
    """Base class for database import operations with proper transaction handling."""

    def __init__(self, batch_size: int = 100):
        """
        Initialize the importer.

        Args:
            batch_size: Number of records to process per batch.
        """
        self.batch_size = batch_size
        self._engine = None
        self._engine_lock = threading.Lock()

    @property
    def engine(self):
        """Thread-safe lazy-load the database engine."""
        if self._engine is None:
            with self._engine_lock:
                if self._engine is None:
                    from sqlalchemy import create_engine
                    self._engine = create_engine(get_sync_db_url())
        return self._engine

    def import_to_postgres(
        self,
        codes: List[Dict[str, Any]],
        on_conflict: str = "skip",
    ) -> tuple[int, int]:
        """
        Import codes to PostgreSQL with proper transaction handling.

        Args:
            codes: List of DTC code dictionaries.
            on_conflict: Behavior on duplicate: "skip", "update", or "error".

        Returns:
            Tuple of (inserted_count, skipped_count).
        """
        from sqlalchemy.orm import Session
        from backend.app.db.postgres.models import Base, DTCCode

        Base.metadata.create_all(self.engine)

        inserted = 0
        skipped = 0

        with Session(self.engine) as session:
            try:
                for i in range(0, len(codes), self.batch_size):
                    batch = codes[i:i + self.batch_size]

                    for code_data in batch:
                        existing = session.query(DTCCode).filter_by(
                            code=code_data["code"]
                        ).first()

                        if existing:
                            if on_conflict == "update":
                                for key, value in code_data.items():
                                    if hasattr(existing, key) and value:
                                        setattr(existing, key, value)
                                inserted += 1
                            else:
                                skipped += 1
                            continue

                        dtc = DTCCode(
                            code=code_data["code"],
                            description_en=code_data["description_en"],
                            description_hu=code_data.get("description_hu"),
                            category=code_data.get("category", "unknown"),
                            severity=code_data.get("severity", "medium"),
                            is_generic=code_data.get("is_generic", True),
                            system=code_data.get("system", ""),
                            symptoms=code_data.get("symptoms", []),
                            possible_causes=code_data.get("possible_causes", []),
                            diagnostic_steps=code_data.get("diagnostic_steps", []),
                            related_codes=code_data.get("related_codes", []),
                        )
                        session.add(dtc)
                        inserted += 1

                    # Commit each batch
                    session.commit()

            except Exception as e:
                session.rollback()
                logger.error(f"PostgreSQL import failed: {e}")
                raise

        return inserted, skipped

    def import_to_neo4j(self, codes: List[Dict[str, Any]]) -> int:
        """
        Import codes to Neo4j.

        Args:
            codes: List of DTC code dictionaries.

        Returns:
            Number of nodes created.
        """
        from backend.app.db.neo4j_models import DTCNode

        created = 0

        for code_data in codes:
            try:
                existing = DTCNode.nodes.get_or_none(code=code_data["code"])

                if not existing:
                    DTCNode(
                        code=code_data["code"],
                        description_en=code_data["description_en"],
                        description_hu=code_data.get("description_hu"),
                        category=code_data.get("category", "unknown"),
                        severity=code_data.get("severity", "medium"),
                        is_generic=code_data.get("is_generic", True),  # Boolean!
                        system=code_data.get("system", ""),
                    ).save()
                    created += 1

            except Exception as e:
                logger.error(f"Neo4j import error for {code_data['code']}: {e}")
                # Continue with other codes

        return created
