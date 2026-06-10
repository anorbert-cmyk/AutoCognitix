"""fix DiagnosisArchive missing FK constraint and ensure indexes idempotently

The 013 migration created `user_id` as a plain UUID with no foreign key
to `users.id`, while the ORM model declares
`ForeignKey("users.id", ondelete="CASCADE")` — so the DB doesn't enforce
the integrity contract the ORM assumes. This migration adds that missing
FK constraint (after purging orphan rows under a SHARE lock).

It also re-asserts the `ix_diagnosis_archive_original_id` and
`ix_diagnosis_archive_user_id` indexes with `if_not_exists=True`. Note that
013's `op.create_table` with `sa.Column(..., index=True)` DOES create these
indexes, so the calls below are normally no-ops — they are an idempotent
safety net for any environment where the indexes are missing anyway.

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
    #    SET LOCAL is transaction-scoped: Alembic runs migrations inside a
    #    transaction, and env.py reuses one connection for all pending
    #    migrations — a session-level SET would leak into later migrations.
    op.execute(sa.text("SET LOCAL lock_timeout = '30s'"))

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

    # 3) Idempotent index safety net: 013 already created these via
    #    sa.Column(..., index=True), so these are normally no-ops;
    #    if_not_exists covers any environment where they are missing.
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
    # raw SQL with IF EXISTS to stay idempotent (tolerates a partial-upgrade
    # rollback where the constraint never landed).
    #
    # The ix_diagnosis_archive_* indexes are intentionally NOT dropped here:
    # they belong to migration 013 (created via sa.Column(..., index=True));
    # the create_index calls in upgrade() are only an idempotent safety net.
    op.execute(
        sa.text(
            "ALTER TABLE diagnosis_archive "
            "DROP CONSTRAINT IF EXISTS diagnosis_archive_user_id_fkey"
        )
    )
