"""Add vehicle_recalls and vehicle_complaints tables.

Revision ID: 003_vehicle_recalls
Revises: 002_add_dtc_sources_column
Create Date: 2026-02-05 14:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '003_vehicle_recalls'
down_revision: Union[str, None] = '002_add_dtc_sources_column'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create vehicle_recalls and vehicle_complaints tables."""

    # 1. vehicle_recalls - NHTSA recall records
    op.create_table(
        'vehicle_recalls',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('campaign_number', sa.String(20), unique=True, index=True, nullable=False),
        sa.Column('nhtsa_id', sa.String(50), nullable=True),
        sa.Column('manufacturer', sa.String(100), nullable=False),
        sa.Column('make', sa.String(50), index=True, nullable=False),
        sa.Column('model', sa.String(100), index=True, nullable=False),
        sa.Column('model_year', sa.Integer(), index=True, nullable=False),
        sa.Column('recall_date', sa.Date(), nullable=True),
        sa.Column('component', sa.String(500), nullable=True),
        sa.Column('summary', sa.Text(), nullable=True),
        sa.Column('consequence', sa.Text(), nullable=True),
        sa.Column('remedy', sa.Text(), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        # Extracted DTC codes from text analysis
        sa.Column('extracted_dtc_codes', postgresql.ARRAY(sa.String()), default=[], nullable=True),
        # Compression: store raw API response as JSONB (compressed)
        sa.Column('raw_response', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
    )

    # Composite indexes for common queries
    op.create_index('ix_vehicle_recalls_make_model_year', 'vehicle_recalls', ['make', 'model', 'model_year'])
    op.create_index('ix_vehicle_recalls_dtc_codes', 'vehicle_recalls', ['extracted_dtc_codes'], postgresql_using='gin')

    # 2. vehicle_complaints - NHTSA complaint records
    op.create_table(
        'vehicle_complaints',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('odi_number', sa.String(20), unique=True, index=True, nullable=False),
        sa.Column('manufacturer', sa.String(100), nullable=False),
        sa.Column('make', sa.String(50), index=True, nullable=False),
        sa.Column('model', sa.String(100), index=True, nullable=False),
        sa.Column('model_year', sa.Integer(), index=True, nullable=False),
        # Incident details
        sa.Column('crash', sa.Boolean(), default=False, nullable=False),
        sa.Column('fire', sa.Boolean(), default=False, nullable=False),
        sa.Column('injuries', sa.Integer(), default=0, nullable=False),
        sa.Column('deaths', sa.Integer(), default=0, nullable=False),
        # Dates
        sa.Column('complaint_date', sa.Date(), nullable=True),
        sa.Column('date_of_incident', sa.Date(), nullable=True),
        # Details
        sa.Column('components', sa.String(500), nullable=True),
        sa.Column('summary', sa.Text(), nullable=True),
        # Extracted DTC codes from text analysis
        sa.Column('extracted_dtc_codes', postgresql.ARRAY(sa.String()), default=[], nullable=True),
        # Compression: store raw API response as JSONB (compressed)
        sa.Column('raw_response', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
    )

    # Composite indexes for common queries
    op.create_index('ix_vehicle_complaints_make_model_year', 'vehicle_complaints', ['make', 'model', 'model_year'])
    op.create_index('ix_vehicle_complaints_dtc_codes', 'vehicle_complaints', ['extracted_dtc_codes'], postgresql_using='gin')
    op.create_index('ix_vehicle_complaints_crash_fire', 'vehicle_complaints', ['crash', 'fire'])

    # 3. dtc_recall_correlations - Links between DTC codes and recalls
    op.create_table(
        'dtc_recall_correlations',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('dtc_code', sa.String(10), sa.ForeignKey('dtc_codes.code'), index=True, nullable=False),
        sa.Column('recall_id', sa.Integer(), sa.ForeignKey('vehicle_recalls.id', ondelete='CASCADE'), index=True, nullable=False),
        sa.Column('confidence', sa.Float(), default=1.0, nullable=False),  # 1.0 = explicit mention, 0.5 = inferred
        sa.Column('extraction_method', sa.String(50), nullable=True),  # 'explicit', 'component_match', 'symptom_match'
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint('dtc_code', 'recall_id', name='uq_dtc_recall_correlation'),
    )

    # 4. dtc_complaint_correlations - Links between DTC codes and complaints
    op.create_table(
        'dtc_complaint_correlations',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('dtc_code', sa.String(10), sa.ForeignKey('dtc_codes.code'), index=True, nullable=False),
        sa.Column('complaint_id', sa.Integer(), sa.ForeignKey('vehicle_complaints.id', ondelete='CASCADE'), index=True, nullable=False),
        sa.Column('confidence', sa.Float(), default=1.0, nullable=False),
        sa.Column('extraction_method', sa.String(50), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint('dtc_code', 'complaint_id', name='uq_dtc_complaint_correlation'),
    )

    # 5. nhtsa_sync_log - Track sync progress for incremental updates
    op.create_table(
        'nhtsa_sync_log',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('make', sa.String(50), index=True, nullable=False),
        sa.Column('model', sa.String(100), nullable=True),  # NULL = all models
        sa.Column('model_year', sa.Integer(), nullable=False),
        sa.Column('data_type', sa.String(20), nullable=False),  # 'recalls' or 'complaints'
        sa.Column('records_synced', sa.Integer(), default=0, nullable=False),
        sa.Column('last_synced_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('sync_status', sa.String(20), default='completed', nullable=False),  # 'completed', 'partial', 'failed'
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.UniqueConstraint('make', 'model', 'model_year', 'data_type', name='uq_nhtsa_sync_log'),
    )


def downgrade() -> None:
    """Drop all NHTSA-related tables in reverse order."""
    op.drop_table('nhtsa_sync_log')
    op.drop_table('dtc_complaint_correlations')
    op.drop_table('dtc_recall_correlations')

    # Drop indexes first
    op.drop_index('ix_vehicle_complaints_crash_fire', table_name='vehicle_complaints')
    op.drop_index('ix_vehicle_complaints_dtc_codes', table_name='vehicle_complaints')
    op.drop_index('ix_vehicle_complaints_make_model_year', table_name='vehicle_complaints')
    op.drop_table('vehicle_complaints')

    op.drop_index('ix_vehicle_recalls_dtc_codes', table_name='vehicle_recalls')
    op.drop_index('ix_vehicle_recalls_make_model_year', table_name='vehicle_recalls')
    op.drop_table('vehicle_recalls')
