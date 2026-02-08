"""Add missing foreign key constraints for data integrity.

Revision ID: 008_add_fk_constraints
Revises: 007_soft_delete
Create Date: 2026-02-08 12:00:00.000000

This migration adds missing foreign key constraint for vehicle_dtc_frequency.engine_code
to ensure referential integrity with vehicle_engines table.
"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '008_add_fk_constraints'
down_revision: Union[str, None] = '007_soft_delete'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add missing foreign key constraints (idempotent)."""

    # Add FK for engine_code in vehicle_dtc_frequency -> vehicle_engines.code
    # This ensures that engine codes referenced in DTC frequency data actually exist
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.table_constraints
                WHERE constraint_name = 'fk_vehicle_dtc_frequency_engine'
                AND table_name = 'vehicle_dtc_frequency'
            ) THEN
                ALTER TABLE vehicle_dtc_frequency
                ADD CONSTRAINT fk_vehicle_dtc_frequency_engine
                FOREIGN KEY (engine_code)
                REFERENCES vehicle_engines (code)
                ON DELETE CASCADE;
            END IF;
        END $$;
    """)


def downgrade() -> None:
    """Remove foreign key constraints."""

    op.execute("""
        ALTER TABLE vehicle_dtc_frequency
        DROP CONSTRAINT IF EXISTS fk_vehicle_dtc_frequency_engine
    """)
