"""
SQL utility functions for safe query construction.

Provides helpers to prevent SQL injection in ILIKE and similar patterns.
"""

import re


def escape_ilike(value: str) -> str:
    """Escape SQL ILIKE special characters (%, _, \\) to prevent wildcard injection.

    Must be applied to all user-supplied values before use in ILIKE patterns.

    Example:
        escaped = escape_ilike(user_input)
        stmt = select(Model).where(Model.name.ilike(f"%{escaped}%"))
    """
    return re.sub(r"([%_\\])", r"\\\1", value)
