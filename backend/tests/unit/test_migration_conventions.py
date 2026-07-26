"""Alembic migration conventions that a reviewer would otherwise have to catch.

`revision` / `down_revision` are read by Alembic through `globals()`
introspection, so every static analyzer reports them as unused globals. The
project's answer (CLAUDE.md; migrations 016/017/018) is the `lgtm[...]`
directive. Migration 020 shipped with `codeql[...]` instead, which is not a
suppression the scanner honours - the annotation looked deliberate while
suppressing nothing.
"""

import ast
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
# visible: the set may shrink, never grow. Now empty - 019 was the last holdout
# and has been converted, so the ratchet is absolute: no migration anywhere may
# carry a codeql[...] comment.
KNOWN_UNCONVERTED: set = set()


def _migration_files() -> list[Path]:
    files = sorted(p for p in VERSIONS_DIR.glob("*.py") if p.name != "__init__.py")
    assert files, f"no migrations found under {VERSIONS_DIR}"
    return files


@pytest.mark.unit
@pytest.mark.parametrize(
    "migration",
    [
        "019_fix_diagnosis_archive_indexes_and_fk.py",
        "020_complaint_component_index.py",
        "021_purge_malformed_dtc_codes.py",
    ],
)
def test_migration_marks_its_revision_ids_with_the_project_directive(migration: str):
    source = (VERSIONS_DIR / migration).read_text(encoding="utf-8")

    for name in ("revision:", "down_revision:"):
        line = next(ln for ln in source.splitlines() if ln.startswith(name))
        assert LGTM.search(line), f"{migration} {name} must carry the lgtm suppression, got: {line}"
        assert not CODEQL.search(line), (
            f"{migration} {name} still carries the inert codeql comment: {line}"
        )


@pytest.mark.unit
def test_no_new_migration_uses_the_inert_codeql_comment():
    """Ratchet: the exception list may shrink, never grow."""
    offenders = {p.name for p in _migration_files() if CODEQL.search(p.read_text(encoding="utf-8"))}

    assert offenders <= KNOWN_UNCONVERTED, (
        f"new migration(s) using the inert codeql[...] comment: "
        f"{sorted(offenders - KNOWN_UNCONVERTED)}. Use lgtm[py/unused-global-variable]."
    )


@pytest.mark.unit
def test_the_migration_chain_has_exactly_one_head():
    """A second head makes `alembic upgrade head` ambiguous and fails a deploy.

    Parsed with `ast` rather than imported: alembic is not a test dependency,
    so `from alembic import op` at the top of every migration would blow up.
    """
    revisions: dict = {}
    parents: set = set()
    for path in _migration_files():
        revision = down = None
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            targets = []
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                targets = [node.target.id]
            elif isinstance(node, ast.Assign):
                targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            for target in targets:
                if target == "revision":
                    revision = ast.literal_eval(node.value)
                elif target == "down_revision":
                    down = ast.literal_eval(node.value)
        if revision:
            revisions[revision] = path.name
        if isinstance(down, str):
            parents.add(down)
        elif isinstance(down, (list, tuple)):
            parents.update(down)

    dangling = parents - set(revisions)
    assert not dangling, f"down_revision pointing at missing migrations: {sorted(dangling)}"

    heads = sorted(rev for rev in revisions if rev not in parents)
    assert len(heads) == 1, f"expected one head, found {heads}"


@pytest.mark.unit
def test_the_dtc_purge_migration_only_targets_codes_the_shared_rule_rejects():
    """021 deletes rows. Its delete list must never contain a real DTC.

    The migration asserts this at runtime too, but it has never been executed
    anywhere, so the runtime guard has never fired. This is the check that
    actually runs.
    """
    from app.core.dtc_codes import is_valid_dtc_code

    source = (VERSIONS_DIR / "021_purge_malformed_dtc_codes.py").read_text(encoding="utf-8")
    codes = None
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "MALFORMED_CODES" for t in node.targets
        ):
            codes = ast.literal_eval(node.value)
    assert codes, "MALFORMED_CODES not found in migration 021"

    accepted = [code for code in codes if is_valid_dtc_code(code)]
    assert not accepted, f"migration 021 would delete valid DTC codes: {accepted}"

    # Same five rows the API tests seed, so the two cannot drift apart.
    assert set(codes) == {"PEACE", "PACED", "P93AF", "UA80E", "UA80F"}


@pytest.mark.unit
def test_the_dtc_purge_migration_deletes_children_before_parents():
    """The FKs to dtc_codes.code carry no ON DELETE, so order is load-bearing.

    Deleting dtc_codes first raises ForeignKeyViolation and aborts the deploy.
    """
    source = (VERSIONS_DIR / "021_purge_malformed_dtc_codes.py").read_text(encoding="utf-8")
    upgrade_body = source.split("def upgrade(")[1].split("def downgrade(")[0]

    child_delete = upgrade_body.index("DELETE FROM {table}")
    parent_delete = upgrade_body.index("DELETE FROM dtc_codes")
    assert child_delete < parent_delete, (
        "migration 021 deletes dtc_codes before its FK children; the correlation "
        "tables reference dtc_codes.code with no ON DELETE action"
    )
