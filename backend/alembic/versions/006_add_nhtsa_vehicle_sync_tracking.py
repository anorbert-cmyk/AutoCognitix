"""Add NHTSA vehicle sync tracking table.

Revision ID: 006_add_nhtsa_vehicle_sync_tracking
Revises: 005_vehicle_comprehensive_schema
Create Date: 2026-02-05 20:00:00.000000

This migration adds a table to track NHTSA vehicle data synchronization progress,
including makes, models, recalls, and complaints sync status.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '006_add_nhtsa_vehicle_sync_tracking'
down_revision: Union[str, None] = '005_vehicle_comprehensive_schema'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create NHTSA vehicle sync tracking table."""

    # Create nhtsa_vehicle_sync_tracking table
    op.create_table(
        'nhtsa_vehicle_sync_tracking',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),

        # What was synced
        sa.Column('sync_type', sa.String(30), nullable=False),  # 'makes', 'models', 'recalls', 'complaints'
        sa.Column('make_name', sa.String(100), nullable=True),  # NULL for 'makes' sync type
        sa.Column('model_name', sa.String(100), nullable=True),  # For model-specific syncs
        sa.Column('year_start', sa.Integer(), nullable=True),
        sa.Column('year_end', sa.Integer(), nullable=True),

        # Sync results
        sa.Column('records_fetched', sa.Integer(), default=0, nullable=False),
        sa.Column('records_saved', sa.Integer(), default=0, nullable=False),
        sa.Column('records_skipped', sa.Integer(), default=0, nullable=False),
        sa.Column('dtc_codes_extracted', sa.Integer(), default=0, nullable=False),

        # Status
        sa.Column('status', sa.String(20), default='completed', nullable=False),  # 'in_progress', 'completed', 'failed', 'partial'
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('error_count', sa.Integer(), default=0, nullable=False),

        # API stats
        sa.Column('api_requests', sa.Integer(), default=0, nullable=False),
        sa.Column('api_errors', sa.Integer(), default=0, nullable=False),
        sa.Column('elapsed_seconds', sa.Float(), nullable=True),

        # Timestamps
        sa.Column('started_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),

        # Extra data (JSONB) - NOTE: 'metadata' is a reserved word in SQLAlchemy!
        sa.Column('sync_metadata', postgresql.JSONB(), nullable=True),  # Additional sync info
    )

    # Create indexes
    op.create_index('ix_nhtsa_vehicle_sync_tracking_type', 'nhtsa_vehicle_sync_tracking', ['sync_type'])
    op.create_index('ix_nhtsa_vehicle_sync_tracking_make', 'nhtsa_vehicle_sync_tracking', ['make_name'])
    op.create_index('ix_nhtsa_vehicle_sync_tracking_status', 'nhtsa_vehicle_sync_tracking', ['status'])
    op.create_index('ix_nhtsa_vehicle_sync_tracking_created', 'nhtsa_vehicle_sync_tracking', ['created_at'])

    # Composite index for looking up specific syncs
    op.create_index(
        'ix_nhtsa_vehicle_sync_tracking_lookup',
        'nhtsa_vehicle_sync_tracking',
        ['sync_type', 'make_name', 'model_name', 'year_start', 'year_end']
    )

    # Add NHTSA-specific fields to vehicle_makes if not exists
    # These track NHTSA API IDs for reference
    op.add_column('vehicle_makes', sa.Column('nhtsa_make_id', sa.Integer(), nullable=True))
    op.create_index('ix_vehicle_makes_nhtsa_id', 'vehicle_makes', ['nhtsa_make_id'])


def downgrade() -> None:
    """Drop NHTSA vehicle sync tracking table and related columns."""

    # Remove NHTSA ID from vehicle_makes
    op.drop_index('ix_vehicle_makes_nhtsa_id', table_name='vehicle_makes')
    op.drop_column('vehicle_makes', 'nhtsa_make_id')

    # Drop indexes
    op.drop_index('ix_nhtsa_vehicle_sync_tracking_lookup', table_name='nhtsa_vehicle_sync_tracking')
    op.drop_index('ix_nhtsa_vehicle_sync_tracking_created', table_name='nhtsa_vehicle_sync_tracking')
    op.drop_index('ix_nhtsa_vehicle_sync_tracking_status', table_name='nhtsa_vehicle_sync_tracking')
    op.drop_index('ix_nhtsa_vehicle_sync_tracking_make', table_name='nhtsa_vehicle_sync_tracking')
    op.drop_index('ix_nhtsa_vehicle_sync_tracking_type', table_name='nhtsa_vehicle_sync_tracking')

    # Drop table
    op.drop_table('nhtsa_vehicle_sync_tracking')
