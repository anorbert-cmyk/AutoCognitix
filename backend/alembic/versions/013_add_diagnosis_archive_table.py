"""Add diagnosis archive table.

Creates an archive table for old diagnosis sessions with JSONB storage
for space efficiency while preserving data for compliance (GDPR).

Revision ID: 013_add_diagnosis_archive_table
Revises: 012_epa_vehicles
Create Date: 2026-03-20
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

# revision identifiers, used by Alembic.
revision: str = "013_add_diagnosis_archive_table"
down_revision: str = "012_epa_vehicles"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "diagnosis_archive",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("original_id", UUID(as_uuid=True), nullable=False, index=True),
        sa.Column("user_id", UUID(as_uuid=True), nullable=False, index=True),
        sa.Column(
            "archived_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("original_created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("session_data", JSONB, nullable=False),
        sa.Column("dtc_codes", JSONB, nullable=True),
        sa.Column("vehicle_info", JSONB, nullable=True),
    )
    # Index for date-range queries on archived sessions
    op.create_index("ix_diagnosis_archive_archived_at", "diagnosis_archive", ["archived_at"])


def downgrade() -> None:
    op.drop_table("diagnosis_archive")
