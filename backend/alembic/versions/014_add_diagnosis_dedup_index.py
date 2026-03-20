"""add composite index for duplicate diagnosis detection

Revision ID: 014_add_diagnosis_dedup_index
Revises: 013_add_diagnosis_archive_table
Create Date: 2026-03-20
"""

from alembic import op

revision: str = "014_add_diagnosis_dedup_index"
down_revision: str = "013_add_diagnosis_archive_table"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Composite index for find_recent_duplicate() query performance
    # Covers: WHERE user_id = ? AND created_at >= ? AND is_deleted = false ORDER BY created_at DESC
    op.create_index(
        "ix_diagnosis_sessions_user_created_active",
        "diagnosis_sessions",
        ["user_id", "created_at", "is_deleted"],
    )


def downgrade() -> None:
    op.drop_index("ix_diagnosis_sessions_user_created_active", table_name="diagnosis_sessions")
