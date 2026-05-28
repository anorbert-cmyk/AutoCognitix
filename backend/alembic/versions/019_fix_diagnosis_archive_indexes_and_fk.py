"""fix DiagnosisArchive missing FK constraint and explicit indexes

The 013 migration declared `index=True` inside `sa.Column(...)` which Alembic
does NOT translate to an `op.create_index` call — so `original_id` and
`user_id` were left without DB indexes despite the ORM model expecting them.
The migration also created `user_id` as a plain UUID with no foreign key
to `users.id`, while the ORM model declares
`ForeignKey("users.id", ondelete="CASCADE")`. Both gaps mean the DB doesn't
enforce the integrity contract the ORM assumes.

Revision ID: 019_fix_archive_drift
Revises: 018_fix_diagnosis_fk
Create Date: 2026-05-28
"""  # lgtm[py/unused-global-variable]

from typing import Union

from alembic import op

revision: str = "019_fix_archive_drift"  # lgtm[py/unused-global-variable]
down_revision: Union[str, None] = "018_fix_diagnosis_fk"  # lgtm[py/unused-global-variable]


def upgrade() -> None:
    # Explicit indexes — Alembic ignored the index=True column flag in 013.
    op.create_index(
        "ix_diagnosis_archive_original_id",
        "diagnosis_archive",
        ["original_id"],
    )
    op.create_index(
        "ix_diagnosis_archive_user_id",
        "diagnosis_archive",
        ["user_id"],
    )

    # FK to users with cascade delete (matches ORM contract for GDPR).
    op.create_foreign_key(
        "diagnosis_archive_user_id_fkey",
        "diagnosis_archive",
        "users",
        ["user_id"],
        ["id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    op.drop_constraint(
        "diagnosis_archive_user_id_fkey",
        "diagnosis_archive",
        type_="foreignkey",
    )
    op.drop_index("ix_diagnosis_archive_user_id", table_name="diagnosis_archive")
    op.drop_index("ix_diagnosis_archive_original_id", table_name="diagnosis_archive")
