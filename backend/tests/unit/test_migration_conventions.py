"""Alembic migration conventions that a reviewer would otherwise have to catch.

`revision` / `down_revision` are read by Alembic through `globals()`
introspection, so every static analyzer reports them as unused globals. The
project's answer (CLAUDE.md; migrations 016/017/018) is the `lgtm[...]`
directive. Migration 020 shipped with `codeql[...]` instead, which is not a
suppression the scanner honours - the annotation looked deliberate while
suppressing nothing.
"""

import re
from pathlib import Path

import pytest

VERSIONS_DIR = Path(__file__).resolve().parents[2] / "alembic" / "versions"

# The directive the project actually uses.
LGTM = re.compile(r"#\s*lgtm\[py/unused-global-variable\]")
# The one that does nothing.
CODEQL = re.compile(r"#\s*codeql\[")

# Pre-existing instances of the same defect, outside the scope of the change
# that added this guard. Listed rather than silently ignored so the ratchet is
# visible: the set may shrink, never grow.
KNOWN_UNCONVERTED = {"019_fix_diagnosis_archive_indexes_and_fk.py"}


def _migration_files() -> list[Path]:
    files = sorted(p for p in VERSIONS_DIR.glob("*.py") if p.name != "__init__.py")
    assert files, f"no migrations found under {VERSIONS_DIR}"
    return files


@pytest.mark.unit
def test_migration_020_marks_its_revision_ids_with_the_project_directive():
    source = (VERSIONS_DIR / "020_complaint_component_index.py").read_text(encoding="utf-8")

    for name in ("revision:", "down_revision:"):
        line = next(ln for ln in source.splitlines() if ln.startswith(name))
        assert LGTM.search(line), f"{name} must carry the lgtm suppression, got: {line}"
        assert not CODEQL.search(line), f"{name} still carries the inert codeql comment: {line}"


@pytest.mark.unit
def test_no_new_migration_uses_the_inert_codeql_comment():
    """Ratchet: the exception list may shrink, never grow."""
    offenders = {p.name for p in _migration_files() if CODEQL.search(p.read_text(encoding="utf-8"))}

    assert offenders <= KNOWN_UNCONVERTED, (
        f"new migration(s) using the inert codeql[...] comment: "
        f"{sorted(offenders - KNOWN_UNCONVERTED)}. Use lgtm[py/unused-global-variable]."
    )
