"""URL validation utilities for web scraping scripts.

This module provides secure URL validation functions to prevent:
- Domain spoofing attacks (e.g., attacker-example.com bypassing example.com validation)
- URL injection attacks
- Path traversal attacks

Security Issue: CWE-20 (Improper Input Validation)
"""

from __future__ import annotations

import re
from urllib.parse import urlparse


def validate_domain(netloc: str, allowed_domain: str) -> bool:
    """
    Validate domain with proper boundary checking.

    This function ensures that the domain matches exactly or is a proper
    subdomain (with dot prefix), preventing attacks like:
    - "attacker-example.com" passing validation for "example.com"
    - "example.com.attacker.com" passing validation

    Args:
        netloc: The network location (domain) from a parsed URL.
        allowed_domain: The allowed domain name (e.g., "example.com").

    Returns:
        True if netloc matches allowed_domain exactly or is a subdomain of it.

    Examples:
        >>> validate_domain("example.com", "example.com")
        True
        >>> validate_domain("sub.example.com", "example.com")
        True
        >>> validate_domain("attacker-example.com", "example.com")
        False
        >>> validate_domain("example.com.attacker.com", "example.com")
        False
    """
    netloc = netloc.lower().strip()
    allowed_domain = allowed_domain.lower().strip()

    # Exact match
    if netloc == allowed_domain:
        return True

    # Subdomain match (must have dot prefix to prevent attacker-domain.com)
    return netloc.endswith("." + allowed_domain)


def validate_scraping_url(url: str, allowed_domains: list[str]) -> bool:
    """
    Validate URL for scraping operations.

    Ensures the URL:
    - Uses HTTP or HTTPS protocol
    - Belongs to one of the allowed domains
    - Does not contain path traversal sequences

    Args:
        url: The URL to validate.
        allowed_domains: List of allowed domain names.

    Returns:
        True if URL is safe to scrape.

    Examples:
        >>> validate_scraping_url("https://example.com/page", ["example.com"])
        True
        >>> validate_scraping_url("https://attacker.com/page", ["example.com"])
        False
        >>> validate_scraping_url("ftp://example.com/page", ["example.com"])
        False
    """
    try:
        parsed = urlparse(url)

        # Must be HTTP(S)
        if parsed.scheme not in ("http", "https"):
            return False

        # Check for path traversal
        if ".." in parsed.path:
            return False

        # Must match an allowed domain
        for domain in allowed_domains:
            if validate_domain(parsed.netloc, domain):
                return True

        return False

    except Exception:
        return False


def validate_dtc_range(dtc_range: str) -> bool:
    """
    Validate DTC range format to prevent injection.

    Expected format: "p0000-p0099", "b1000-b1999", etc.
    This prevents query parameter injection attacks.

    Args:
        dtc_range: The DTC range string to validate.

    Returns:
        True if the format is valid.

    Examples:
        >>> validate_dtc_range("p0000-p0099")
        True
        >>> validate_dtc_range("P0300-P0399")
        True
        >>> validate_dtc_range("p0000&evil=injection")
        False
        >>> validate_dtc_range("../../../etc/passwd")
        False
    """
    # Valid format: letter + 4 digits + hyphen + letter + 4 digits
    pattern = r"^[pbcu]\d{4}-[pbcu]\d{4}$"
    return bool(re.match(pattern, dtc_range, re.IGNORECASE))


def sanitize_path(path: str) -> str:
    """
    Remove path traversal attempts from a path string.

    This function removes ".." sequences that could be used for
    directory traversal attacks.

    Args:
        path: The path string to sanitize.

    Returns:
        Sanitized path string.

    Examples:
        >>> sanitize_path("/normal/path")
        '/normal/path'
        >>> sanitize_path("../../../etc/passwd")
        'etc/passwd'
        >>> sanitize_path("/path/../to/../file")
        '/path/to/file'
    """
    # Remove all occurrences of ".."
    while ".." in path:
        path = path.replace("..", "")

    # Clean up any resulting double slashes
    while "//" in path:
        path = path.replace("//", "/")

    return path


def is_valid_obd_codes_url(url: str) -> bool:
    """
    Validate URL for obd-codes.com scraping.

    Args:
        url: URL to validate.

    Returns:
        True if URL belongs to obd-codes.com domain.
    """
    return validate_scraping_url(url, ["obd-codes.com", "www.obd-codes.com"])


def is_valid_engine_codes_url(url: str) -> bool:
    """
    Validate URL for engine-codes.com scraping.

    Args:
        url: URL to validate.

    Returns:
        True if URL belongs to engine-codes.com domain.
    """
    return validate_scraping_url(url, ["engine-codes.com", "www.engine-codes.com"])


def is_valid_klavkarr_url(url: str) -> bool:
    """
    Validate URL for klavkarr.com scraping.

    Args:
        url: URL to validate.

    Returns:
        True if URL belongs to klavkarr.com domain.
    """
    return validate_scraping_url(url, ["klavkarr.com", "www.klavkarr.com"])
