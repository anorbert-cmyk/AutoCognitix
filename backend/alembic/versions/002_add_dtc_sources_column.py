"""Add sources column to dtc_codes table.

Revision ID: 002_add_dtc_sources_column
Revises: 001_initial_schema
Create Date: 2026-02-05 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '002_add_dtc_sources_column'
down_revision: Union[str, None] = '001_initial_schema'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add sources column to dtc_codes table for tracking data origins."""
    op.add_column(
        'dtc_codes',
        sa.Column('sources', postgresql.ARRAY(sa.String()), nullable=True, default=[])
    )


def downgrade() -> None:
    """Remove sources column from dtc_codes table."""
    op.drop_column('dtc_codes', 'sources')
