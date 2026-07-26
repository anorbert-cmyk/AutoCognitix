"""purge the five malformed DTC rows the old loose pattern imported

!!! NOT RUN ANYWHERE !!!
------------------------
This migration has NOT been executed against any database - not production, not
staging, not a local instance. It was written from the schema, and the row
counts it reports are unknown until someone runs it. Treat the first run as a
change requiring a backup and a read of the emitted audit lines.

What it removes
---------------
`dtc_codes` contains five rows that are not diagnostic trouble codes:

    PEACE, PACED, P93AF, UA80E, UA80F

They entered the corpus through the historical `[PBCU][0-9A-F]{4}` pattern,
which accepts hex-shaped English words and manufacturer designations because it
never constrained the second character. `app.core.dtc_codes` now enforces the
SAE J2012 rule (second character `0-3`) and `DTCCreate` validates writes
against it, so the door is shut - these rows are the mess already inside.

They are unreachable through the API: `/dtc/search` filters results through
`_servable_dtcs`, and `/dtc/{code}` answers 400 for them. So this is a cleanup,
not a bug fix, and there is no urgency to run it. What it does buy: any direct
DB consumer, export, count or future re-index stops seeing five fake codes, and
`related_codes` arrays stop pointing at rows that cannot be opened.

Why the dependent tables are handled first
------------------------------------------
`vehicle_dtc_frequency`, `dtc_recall_correlations` and `dtc_complaint_correlations`
all carry `dtc_code -> dtc_codes.code` foreign keys declared WITHOUT an
`ondelete`, so PostgreSQL defaults to NO ACTION. A bare
`DELETE FROM dtc_codes WHERE code IN (...)` would therefore abort the whole
migration - and the deploy - the moment any correlation row references one of
the five. Children go first, in FK order.

Deliberately NOT a rule-driven delete
-------------------------------------
The delete list is five literals, not "every row failing `is_valid_dtc_code`".
A migration that deletes an unbounded, rule-computed set is a migration whose
blast radius changes whenever the rule is tuned. The shared rule IS used, but
only to (a) assert the five literals really are invalid, so a later edit cannot
smuggle a real code into the list, and (b) REPORT any other malformed rows
without touching them, so an operator learns about them from the deploy log and
decides deliberately.

Irreversible
------------
`downgrade()` is a documented no-op: the rows cannot be resurrected from within
the migration. Rolling back leaves the table clean, which is the desired state
anyway.

Revision ID: 021_purge_malformed_dtc
Revises: 020_complaint_component_idx
Create Date: 2026-07-26
"""

from typing import Union

import sqlalchemy as sa
from alembic import op

from app.core.dtc_codes import is_valid_dtc_code

# revision identifiers, used by Alembic at runtime via globals() introspection.
# The static analyzer can't see that usage, so we mark them explicitly with the
# `lgtm[...]` directive this project uses everywhere else (CLAUDE.md; migrations
# 016/017/018/019/020).
revision: str = "021_purge_malformed_dtc"  # lgtm[py/unused-global-variable]
down_revision: Union[str, None] = "020_complaint_component_idx"  # lgtm[py/unused-global-variable]

__all__ = ["revision", "down_revision", "upgrade", "downgrade"]

# The exact rows documented in CLAUDE.md and pinned by
# tests/api/test_dtc.py::SEEDED_JUNK_CODES.
MALFORMED_CODES = ("PEACE", "PACED", "P93AF", "UA80E", "UA80F")

# Tables whose `dtc_code` column references dtc_codes.code with no ON DELETE
# action. Ordered children-first; all three are siblings, so any order works,
# but they must all precede the parent delete.
DEPENDENT_TABLES = (
    "vehicle_dtc_frequency",
    "dtc_recall_correlations",
    "dtc_complaint_correlations",
)


def _table_exists(bind: sa.engine.Connection, table: str) -> bool:
    """True if `table` is present in the current search_path.

    Environments seeded from different points in the migration history do not
    all have every correlation table; a missing one must be skipped, not crash
    a cleanup that is meant to be safe to run late.
    """
    return bind.execute(sa.text("SELECT to_regclass(:t)"), {"t": table}).scalar() is not None


def upgrade() -> None:
    bind = op.get_bind()

    # 0) Fail closed if the literal list ever drifts into accepting a real code.
    #    Cheap, and it is the only thing standing between this migration and an
    #    edit that quietly deletes P0300.
    real = [code for code in MALFORMED_CODES if is_valid_dtc_code(code)]
    if real:
        raise RuntimeError(
            f"migration 021 refuses to delete structurally VALID DTC codes: {real}. "
            "The delete list must contain only codes app.core.dtc_codes rejects."
        )

    if not _table_exists(bind, "dtc_codes"):
        print("[migration 021] dtc_codes table absent - nothing to purge")
        return

    codes = list(MALFORMED_CODES)

    # 1) Children first: the FKs have no ON DELETE, so the parent delete below
    #    would raise ForeignKeyViolation while any of these rows survive.
    #    The table NAME is interpolated because SQL identifiers cannot be bound;
    #    it comes from the DEPENDENT_TABLES constant above and never from input.
    #    Every VALUE is a bound parameter.
    for table in DEPENDENT_TABLES:
        if not _table_exists(bind, table):
            print(f"[migration 021] {table} absent - skipped")
            continue
        result = bind.execute(
            sa.text(
                f"DELETE FROM {table} "
                "WHERE dtc_code = ANY(CAST(:codes AS varchar[])) RETURNING id"
            ),
            {"codes": codes},
        )
        removed = [row[0] for row in result]
        if removed:
            print(f"[migration 021] {table}: deleted {len(removed)} rows referencing malformed DTCs")

    # 2) Drop the malformed codes out of every surviving related_codes array, so
    #    no readable row keeps recommending a code that no longer exists. Chained
    #    array_remove is idempotent, each removed value is a bound parameter, and
    #    the WHERE clause touches only rows that actually contain one of them.
    scrub = "related_codes"
    params = {"codes": codes}
    for i, code in enumerate(codes):
        params[f"c{i}"] = code
        scrub = f"array_remove({scrub}, :c{i})"
    result = bind.execute(
        sa.text(
            f"UPDATE dtc_codes SET related_codes = {scrub} "
            "WHERE related_codes && CAST(:codes AS varchar[]) RETURNING code"
        ),
        params,
    )
    scrubbed = [row[0] for row in result]
    if scrubbed:
        print(
            f"[migration 021] scrubbed malformed entries from related_codes on "
            f"{len(scrubbed)} rows: {scrubbed[:25]}"
        )

    # 3) The parent rows.
    result = bind.execute(
        sa.text("DELETE FROM dtc_codes WHERE code = ANY(CAST(:codes AS varchar[])) RETURNING code"),
        {"codes": codes},
    )
    purged = sorted(row[0] for row in result)
    print(f"[migration 021] purged {len(purged)} malformed dtc_codes rows: {purged}")

    # 4) Report-only: anything else that fails the shared rule is left in place
    #    and surfaced in the deploy log. Deleting it here would make this
    #    migration's blast radius a function of a rule that may still be tuned.
    remaining = [
        row[0]
        for row in bind.execute(sa.text("SELECT code FROM dtc_codes"))
        if not is_valid_dtc_code(row[0])
    ]
    if remaining:
        print(
            f"[migration 021] NOT DELETED - {len(remaining)} further rows fail the "
            f"SAE J2012 rule and need a deliberate decision: {sorted(remaining)[:50]}"
        )


def downgrade() -> None:
    # Intentionally a no-op. The deleted rows were never valid DTCs and cannot
    # be reconstructed from within a migration; restoring them would mean
    # re-importing the corpus dump they came from. Downgrading past this point
    # simply leaves the table in the cleaned state.
    pass
