"""fix DiagnosisArchive missing FK constraint and explicit indexes

The 013 migration declared `index=True` inside `sa.Column(...)` which Alembic
does NOT translate to an `op.create_index` call — so `original_id` and
`user_id` were left without DB indexes despite the ORM model expecting them.
The migration also created `user_id` as a plain UUID with no foreign key
to `users.id`, while the ORM model declares
`ForeignKey("users.id", ondelete="CASCADE")`. Both gaps mean the DB doesn't
enforce the integrity contract the ORM assumes.

Production safety notes:
- The table is briefly locked in SHARE mode to prevent concurrent INSERTs
  from creating new orphan rows between the purge and the FK validation
  (TOCTOU race that would crash the deploy).
- Orphan rows (user_id pointing to a since-deleted user) are deleted BEFORE
  the FK constraint is added; their IDs are RETURNINGed and printed to the
  Alembic stdout so the deploy log preserves an audit trail (GDPR).
- The archive table is small in practice, so the SHARE lock for the duration
  of the migration is acceptable. For very large tables a separate online
  pattern (NOT VALID + VALIDATE in two transactions) would be required.

Revision ID: 019_fix_archive_drift
Revises: 018_fix_diagnosis_fk
Create Date: 2026-05-28
"""

from typing import Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic at runtime via globals() introspection.
# CodeQL's static analyzer can't see that usage, so we mark them explicitly.
revision: str = "019_fix_archive_drift"  # noqa: F841  # codeql[py/unused-global-variable]
down_revision: Union[str, None] = "018_fix_diagnosis_fk"  # noqa: F841  # codeql[py/unused-global-variable]

__all__ = ["revision", "down_revision", "upgrade", "downgrade"]


def upgrade() -> None:
    bind = op.get_bind()

    # 1) Cap the lock acquisition time so a pathologically slow migration
    #    can't starve the app's connection pool on production. 30s is well
    #    above expected migration time for this small table; tune up if the
    #    archive grows past millions of rows.
    op.execute(sa.text("SET lock_timeout = '30s'"))

    # 2) Lock the table briefly to prevent concurrent INSERTs from creating
    #    new orphan rows in the window between purge and FK validation.
    #    SHARE mode blocks writes but allows other concurrent reads.
    op.execute(sa.text("LOCK TABLE diagnosis_archive IN SHARE MODE"))

    # 2) Purge orphan rows and audit-log their IDs (GDPR / forensics).
    #    `NOT IN` is NULL-safe here thanks to the explicit `IS NOT NULL`
    #    guard — without it, any NULL in the users subquery would make
    #    the whole NOT IN return UNKNOWN and skip the purge entirely.
    result = bind.execute(
        sa.text(
            "DELETE FROM diagnosis_archive "
            "WHERE user_id IS NOT NULL "
            "AND user_id NOT IN (SELECT id FROM users) "
            "RETURNING id"
        )
    )
    purged = [str(row[0]) for row in result]
    if purged:
        # Alembic stdout is captured by CI/CD logs — keeps a trail. Print the
        # full count plus head/tail samples; dumping all IDs blows past per-line
        # log limits (~4-64 KB on Railway / GitHub Actions) and ironically
        # truncates the audit trail we're trying to preserve.
        sample_head = purged[:25]
        sample_tail = purged[-25:] if len(purged) > 50 else []
        print(
            f"[migration 019] purged {len(purged)} orphan diagnosis_archive rows "
            f"(first 25: {sample_head}"
            + (f", last 25: {sample_tail}" if sample_tail else "")
            + ")"
        )

    # 3) Explicit indexes (idempotent on retry).
    op.create_index(
        "ix_diagnosis_archive_original_id",
        "diagnosis_archive",
        ["original_id"],
        if_not_exists=True,
    )
    op.create_index(
        "ix_diagnosis_archive_user_id",
        "diagnosis_archive",
        ["user_id"],
        if_not_exists=True,
    )

    # 4) FK with cascade delete (matches the ORM contract for GDPR).
    #    Under the SHARE lock no new orphans can land between purge and
    #    constraint creation, so a plain ADD CONSTRAINT is safe — we do
    #    not need the NOT VALID/VALIDATE split.
    op.create_foreign_key(
        "diagnosis_archive_user_id_fkey",
        "diagnosis_archive",
        "users",
        ["user_id"],
        ["id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    # NOTE: downgrade leaves the table without FK enforcement; a subsequent
    # re-upgrade WILL silently purge any new orphans created in the interim.
    #
    # Alembic's op.drop_constraint doesn't accept an if_exists flag, so we use
    # raw SQL with IF EXISTS to stay idempotent (matches the indexes below and
    # tolerates a partial-upgrade rollback where the constraint never landed).
    op.execute(
        sa.text(
            "ALTER TABLE diagnosis_archive "
            "DROP CONSTRAINT IF EXISTS diagnosis_archive_user_id_fkey"
        )
    )
    op.drop_index("ix_diagnosis_archive_user_id", table_name="diagnosis_archive", if_exists=True)
    op.drop_index(
        "ix_diagnosis_archive_original_id", table_name="diagnosis_archive", if_exists=True
    )
