"""
SQLite compatibility patches for PostgreSQL-specific SQLAlchemy types.

Patches ARRAY and JSONB type compilation and result processing so that
tests can use an in-memory SQLite database instead of PostgreSQL.

Usage: call ``apply_sqlite_patches()`` at module level in each conftest
that creates a SQLite test engine.
"""

from __future__ import annotations

import json
import sqlite3

from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.dialects.sqlite import base as sqlite_base


def apply_sqlite_patches() -> None:
    """Apply all SQLite compatibility patches (idempotent)."""
    if getattr(apply_sqlite_patches, "_done", False):
        return
    apply_sqlite_patches._done = True

    # -- Type compilation: render ARRAY / JSONB as JSON, UUID as CHAR in SQLite

    def _visit_ARRAY(self, type_, **kw):
        return "JSON"

    def _visit_JSONB(self, type_, **kw):
        return "JSON"

    def _visit_UUID(self, type_, **kw):
        return "CHAR(36)"

    sqlite_base.SQLiteTypeCompiler.visit_ARRAY = _visit_ARRAY
    sqlite_base.SQLiteTypeCompiler.visit_JSONB = _visit_JSONB
    sqlite_base.SQLiteTypeCompiler.visit_UUID = _visit_UUID

    # -- Result processors: deserialise JSON strings back to Python objects --

    _orig_array_result_processor = ARRAY.result_processor

    def _array_result_processor(self, dialect, coltype):
        if dialect.name == "sqlite":

            def process(value):
                if value is None:
                    return value
                if isinstance(value, str):
                    return json.loads(value)
                return value

            return process
        return _orig_array_result_processor(self, dialect, coltype)

    ARRAY.result_processor = _array_result_processor

    _orig_jsonb_result_processor = JSONB.result_processor

    def _jsonb_result_processor(self, dialect, coltype):
        if dialect.name == "sqlite":

            def process(value):
                if value is None:
                    return value
                if isinstance(value, str):
                    return json.loads(value)
                return value

            return process
        if _orig_jsonb_result_processor:
            return _orig_jsonb_result_processor(self, dialect, coltype)
        return None

    JSONB.result_processor = _jsonb_result_processor

    # -- SQLite adapters: store Python lists/dicts as JSON strings ----------

    sqlite3.register_adapter(list, json.dumps)
    sqlite3.register_adapter(dict, json.dumps)
