"""Add soft delete columns to diagnosis_sessions and updated_at.

Revision ID: 007_soft_delete
Revises: 006_nhtsa_sync
Create Date: 2026-02-05 22:00:00.000000

This migration adds soft delete support and missing columns to diagnosis_sessions table.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '007_soft_delete'
down_revision: Union[str, None] = '006_nhtsa_sync'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add soft delete columns to diagnosis_sessions."""

    # Add is_deleted column (idempotent)
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                           WHERE table_name='diagnosis_sessions' AND column_name='is_deleted') THEN
                ALTER TABLE diagnosis_sessions ADD COLUMN is_deleted BOOLEAN DEFAULT FALSE NOT NULL;
            END IF;
        END $$;
    """)
    op.execute('CREATE INDEX IF NOT EXISTS ix_diagnosis_sessions_is_deleted ON diagnosis_sessions (is_deleted)')

    # Add deleted_at column (idempotent)
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                           WHERE table_name='diagnosis_sessions' AND column_name='deleted_at') THEN
                ALTER TABLE diagnosis_sessions ADD COLUMN deleted_at TIMESTAMP WITH TIME ZONE;
            END IF;
        END $$;
    """)

    # Add updated_at column (idempotent) - was missing from initial schema
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                           WHERE table_name='diagnosis_sessions' AND column_name='updated_at') THEN
                ALTER TABLE diagnosis_sessions ADD COLUMN updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL;
            END IF;
        END $$;
    """)


def downgrade() -> None:
    """Remove soft delete columns from diagnosis_sessions."""

    op.execute('DROP INDEX IF EXISTS ix_diagnosis_sessions_is_deleted')
    op.execute('ALTER TABLE diagnosis_sessions DROP COLUMN IF EXISTS is_deleted')
    op.execute('ALTER TABLE diagnosis_sessions DROP COLUMN IF EXISTS deleted_at')
    op.execute('ALTER TABLE diagnosis_sessions DROP COLUMN IF EXISTS updated_at')
