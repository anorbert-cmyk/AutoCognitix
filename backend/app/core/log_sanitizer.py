"""Log sanitization utilities to prevent log injection attacks.

This module provides functions to sanitize user input before logging,
preventing attackers from injecting malicious log entries via newlines,
carriage returns, or other control characters.

Security Issue: CWE-117 (Improper Output Neutralization for Logs)
"""

import re
from typing import Any, Dict


def sanitize_log(value: Any, max_length: int = 200) -> str:
    """Sanitize user input for safe logging.

    This function removes or escapes dangerous characters that could be used
    for log injection attacks, including:
    - Newlines (\\n) - could create fake log entries
    - Carriage returns (\\r) - could overwrite log lines
    - Tabs (\\t) - could misalign log formatting
    - Other control characters (ASCII 0-31)

    Args:
        value: The value to sanitize. Can be any type, will be converted to string.
        max_length: Maximum length of the output string. Default 200.

    Returns:
        A sanitized string safe for logging.

    Example:
        >>> sanitize_log("P0101\\nCRITICAL: hacked")
        'P0101\\\\nCRITICAL: hacked'

        >>> sanitize_log(None)
        '[None]'
    """
    if value is None:
        return "[None]"

    # Convert to string if needed
    text = str(value)

    # Replace newlines and carriage returns with escaped versions
    # This makes them visible in logs rather than creating new lines
    text = text.replace("\n", "\\n")
    text = text.replace("\r", "\\r")
    text = text.replace("\t", "\\t")

    # Remove other control characters (ASCII 0-31 except already handled)
    # \x00-\x08: NULL through BACKSPACE
    # \x0b: Vertical tab
    # \x0c: Form feed
    # \x0e-\x1f: Shift out through Unit separator
    # \x7f: DEL
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Truncate if too long to prevent log flooding
    if len(text) > max_length:
        text = text[:max_length] + "...[truncated]"

    return text


def sanitize_exception(exc: BaseException, max_length: int = 500) -> str:
    """Sanitize exception message for logging.

    Exceptions may contain user input in their messages, so they need
    to be sanitized before logging.

    Args:
        exc: The exception to sanitize.
        max_length: Maximum length of the output string. Default 500.

    Returns:
        A sanitized string representation of the exception.

    Example:
        >>> try:
        ...     raise ValueError("Invalid input: test\\nevil")
        ... except ValueError as e:
        ...     sanitize_exception(e)
        'Invalid input: test\\\\nevil'
    """
    return sanitize_log(str(exc), max_length)


def sanitize_dict_values(data: Dict[str, Any], max_length: int = 200) -> Dict[str, str]:
    """Sanitize all values in a dictionary for logging.

    Useful when logging request parameters or other dictionaries
    that may contain user input.

    Args:
        data: Dictionary with potentially unsafe values.
        max_length: Maximum length for each value. Default 200.

    Returns:
        A new dictionary with all values sanitized.

    Example:
        >>> sanitize_dict_values({"q": "test\\nevil", "page": 1})
        {'q': 'test\\\\nevil', 'page': '1'}
    """
    return {str(k): sanitize_log(v, max_length) for k, v in data.items()}
