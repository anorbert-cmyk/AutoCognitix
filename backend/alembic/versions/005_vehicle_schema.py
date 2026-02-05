"""Add comprehensive vehicle schema with engines, platforms, and DTC frequency.

Revision ID: 005_vehicle_schema
Revises: 004_perf_indexes
Create Date: 2026-02-05 16:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '005_vehicle_schema'
down_revision: Union[str, None] = '004_perf_indexes'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create comprehensive vehicle schema tables."""

    # 1. vehicle_engines - Engine specifications
    op.create_table(
        'vehicle_engines',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('code', sa.String(30), unique=True, index=True, nullable=False),
        sa.Column('name', sa.String(200), nullable=True),
        # Engine specifications
        sa.Column('displacement_cc', sa.Integer(), nullable=True),  # Displacement in cubic centimeters
        sa.Column('displacement_l', sa.Float(), nullable=True),  # Displacement in liters
        sa.Column('cylinders', sa.Integer(), nullable=True),
        sa.Column('configuration', sa.String(30), nullable=True),  # inline, v, boxer, rotary
        sa.Column('fuel_type', sa.String(30), index=True, nullable=False),  # gasoline, diesel, hybrid, electric, lpg, cng
        sa.Column('aspiration', sa.String(30), nullable=True),  # naturally_aspirated, turbo, supercharged, twin_turbo
        # Power output
        sa.Column('power_hp', sa.Integer(), nullable=True),
        sa.Column('power_kw', sa.Integer(), nullable=True),
        sa.Column('torque_nm', sa.Integer(), nullable=True),
        # Additional specs
        sa.Column('valves_per_cylinder', sa.Integer(), nullable=True),
        sa.Column('compression_ratio', sa.String(20), nullable=True),
        sa.Column('bore_mm', sa.Float(), nullable=True),
        sa.Column('stroke_mm', sa.Float(), nullable=True),
        # Manufacturer info
        sa.Column('manufacturer', sa.String(100), nullable=True),
        sa.Column('family', sa.String(100), nullable=True),  # Engine family (e.g., EA888, N54, M54)
        # Production years
        sa.Column('year_start', sa.Integer(), nullable=True),
        sa.Column('year_end', sa.Integer(), nullable=True),
        # Applicable makes (for filtering)
        sa.Column('applicable_makes', postgresql.ARRAY(sa.String()), default=[], nullable=True),
        # Metadata
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
    )

    # Indexes for vehicle_engines
    # Note: fuel_type already has index=True in column definition, so we skip explicit create
    op.execute('CREATE INDEX IF NOT EXISTS ix_vehicle_engines_family ON vehicle_engines (family)')
    op.execute('CREATE INDEX IF NOT EXISTS ix_vehicle_engines_displacement ON vehicle_engines (displacement_cc)')
    op.execute('CREATE INDEX IF NOT EXISTS ix_vehicle_engines_applicable_makes ON vehicle_engines USING GIN (applicable_makes)')

    # 2. vehicle_platforms - Shared platforms across makes
    op.create_table(
        'vehicle_platforms',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('code', sa.String(50), unique=True, index=True, nullable=False),
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('manufacturer', sa.String(100), nullable=True),  # Platform owner
        # Shared across makes
        sa.Column('makes', postgresql.ARRAY(sa.String()), default=[], nullable=False),
        # Production years
        sa.Column('year_start', sa.Integer(), nullable=True),
        sa.Column('year_end', sa.Integer(), nullable=True),
        # Platform details
        sa.Column('segment', sa.String(50), nullable=True),  # A, B, C, D, E, F segments
        sa.Column('body_types', postgresql.ARRAY(sa.String()), default=[], nullable=True),
        sa.Column('drivetrain_options', postgresql.ARRAY(sa.String()), default=[], nullable=True),  # FWD, RWD, AWD
        # Compatible engine codes
        sa.Column('compatible_engines', postgresql.ARRAY(sa.String()), default=[], nullable=True),
        # Metadata
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
    )

    # Indexes for vehicle_platforms
    op.execute('CREATE INDEX IF NOT EXISTS ix_vehicle_platforms_makes ON vehicle_platforms USING GIN (makes)')
    op.execute('CREATE INDEX IF NOT EXISTS ix_vehicle_platforms_segment ON vehicle_platforms (segment)')

    # 3. vehicle_model_engines - Many-to-many: models to engines
    op.create_table(
        'vehicle_model_engines',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('model_id', sa.String(50), sa.ForeignKey('vehicle_models.id', ondelete='CASCADE'), nullable=False),
        sa.Column('engine_id', sa.Integer(), sa.ForeignKey('vehicle_engines.id', ondelete='CASCADE'), nullable=False),
        # Production years for this combination
        sa.Column('year_start', sa.Integer(), nullable=True),
        sa.Column('year_end', sa.Integer(), nullable=True),
        # Variant info
        sa.Column('variant_name', sa.String(100), nullable=True),  # e.g., "2.0 TSI 190HP"
        sa.Column('is_base', sa.Boolean(), default=False, nullable=False),  # Is this the base engine option
        sa.UniqueConstraint('model_id', 'engine_id', 'year_start', name='uq_model_engine_year'),
    )

    # Indexes for vehicle_model_engines
    op.execute('CREATE INDEX IF NOT EXISTS ix_vehicle_model_engines_model ON vehicle_model_engines (model_id)')
    op.execute('CREATE INDEX IF NOT EXISTS ix_vehicle_model_engines_engine ON vehicle_model_engines (engine_id)')

    # 4. vehicle_dtc_frequency - Which DTCs are common for which vehicles
    op.create_table(
        'vehicle_dtc_frequency',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('dtc_code', sa.String(10), sa.ForeignKey('dtc_codes.code'), index=True, nullable=False),
        # Vehicle reference (can be make-level, model-level, or engine-level)
        sa.Column('make_id', sa.String(50), sa.ForeignKey('vehicle_makes.id', ondelete='CASCADE'), index=True, nullable=True),
        sa.Column('model_id', sa.String(50), sa.ForeignKey('vehicle_models.id', ondelete='CASCADE'), index=True, nullable=True),
        sa.Column('engine_code', sa.String(30), index=True, nullable=True),
        # Year range
        sa.Column('year_start', sa.Integer(), nullable=True),
        sa.Column('year_end', sa.Integer(), nullable=True),
        # Frequency data
        sa.Column('frequency', sa.String(30), default='common', nullable=False),  # rare, uncommon, common, very_common
        sa.Column('occurrence_count', sa.Integer(), default=0, nullable=False),  # Number of reported occurrences
        sa.Column('confidence', sa.Float(), default=0.5, nullable=False),  # 0.0-1.0
        # Source info
        sa.Column('source', sa.String(50), nullable=True),  # nhtsa, tsb, forum, user_reports
        sa.Column('source_url', sa.String(500), nullable=True),
        # Common symptoms/issues when this DTC appears for this vehicle
        sa.Column('common_symptoms', postgresql.ARRAY(sa.String()), default=[], nullable=True),
        sa.Column('common_causes', postgresql.ARRAY(sa.String()), default=[], nullable=True),
        sa.Column('known_fixes', postgresql.ARRAY(sa.String()), default=[], nullable=True),
        # Metadata
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        # Constraint: at least one vehicle reference must be provided
        sa.CheckConstraint('make_id IS NOT NULL OR model_id IS NOT NULL OR engine_code IS NOT NULL', name='ck_vehicle_dtc_frequency_vehicle_ref'),
    )

    # Indexes for vehicle_dtc_frequency
    op.execute('CREATE INDEX IF NOT EXISTS ix_vehicle_dtc_frequency_dtc ON vehicle_dtc_frequency (dtc_code)')
    op.execute('CREATE INDEX IF NOT EXISTS ix_vehicle_dtc_frequency_make_model ON vehicle_dtc_frequency (make_id, model_id)')
    op.execute('CREATE INDEX IF NOT EXISTS ix_vehicle_dtc_frequency_engine ON vehicle_dtc_frequency (engine_code)')
    op.execute('CREATE INDEX IF NOT EXISTS ix_vehicle_dtc_frequency_frequency ON vehicle_dtc_frequency (frequency)')

    # 5. Update vehicle_models to add foreign key to platform (if not exists)
    # Use raw SQL for idempotent column addition
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                           WHERE table_name='vehicle_models' AND column_name='platform_id') THEN
                ALTER TABLE vehicle_models ADD COLUMN platform_id INTEGER REFERENCES vehicle_platforms(id) ON DELETE SET NULL;
            END IF;
        END $$;
    """)
    op.execute('CREATE INDEX IF NOT EXISTS ix_vehicle_models_platform ON vehicle_models (platform_id)')

    # 6. vehicle_tsb - Technical Service Bulletins
    op.create_table(
        'vehicle_tsb',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('bulletin_number', sa.String(50), unique=True, index=True, nullable=False),
        sa.Column('title', sa.String(500), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        # Vehicle applicability
        sa.Column('make_id', sa.String(50), sa.ForeignKey('vehicle_makes.id', ondelete='CASCADE'), index=True, nullable=True),
        sa.Column('applicable_models', postgresql.ARRAY(sa.String()), default=[], nullable=True),
        sa.Column('year_start', sa.Integer(), nullable=True),
        sa.Column('year_end', sa.Integer(), nullable=True),
        # TSB details
        sa.Column('issue_date', sa.Date(), nullable=True),
        sa.Column('component', sa.String(200), nullable=True),
        sa.Column('related_dtc_codes', postgresql.ARRAY(sa.String()), default=[], nullable=True),
        # Source
        sa.Column('source', sa.String(100), nullable=True),
        sa.Column('source_url', sa.String(500), nullable=True),
        # Metadata
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
    )

    # Indexes for vehicle_tsb
    op.execute('CREATE INDEX IF NOT EXISTS ix_vehicle_tsb_make ON vehicle_tsb (make_id)')
    op.execute('CREATE INDEX IF NOT EXISTS ix_vehicle_tsb_dtc_codes ON vehicle_tsb USING GIN (related_dtc_codes)')


def downgrade() -> None:
    """Drop all vehicle-related tables in reverse order."""
    # Drop TSB table
    op.drop_index('ix_vehicle_tsb_dtc_codes', table_name='vehicle_tsb')
    op.drop_index('ix_vehicle_tsb_make', table_name='vehicle_tsb')
    op.drop_table('vehicle_tsb')

    # Remove platform_id from vehicle_models
    op.drop_index('ix_vehicle_models_platform', table_name='vehicle_models')
    op.drop_column('vehicle_models', 'platform_id')

    # Drop vehicle_dtc_frequency
    op.drop_index('ix_vehicle_dtc_frequency_frequency', table_name='vehicle_dtc_frequency')
    op.drop_index('ix_vehicle_dtc_frequency_engine', table_name='vehicle_dtc_frequency')
    op.drop_index('ix_vehicle_dtc_frequency_make_model', table_name='vehicle_dtc_frequency')
    op.drop_index('ix_vehicle_dtc_frequency_dtc', table_name='vehicle_dtc_frequency')
    op.drop_table('vehicle_dtc_frequency')

    # Drop vehicle_model_engines
    op.drop_index('ix_vehicle_model_engines_engine', table_name='vehicle_model_engines')
    op.drop_index('ix_vehicle_model_engines_model', table_name='vehicle_model_engines')
    op.drop_table('vehicle_model_engines')

    # Drop vehicle_platforms
    op.drop_index('ix_vehicle_platforms_segment', table_name='vehicle_platforms')
    op.drop_index('ix_vehicle_platforms_makes', table_name='vehicle_platforms')
    op.drop_table('vehicle_platforms')

    # Drop vehicle_engines
    op.drop_index('ix_vehicle_engines_applicable_makes', table_name='vehicle_engines')
    op.drop_index('ix_vehicle_engines_displacement', table_name='vehicle_engines')
    op.drop_index('ix_vehicle_engines_family', table_name='vehicle_engines')
    op.drop_index('ix_vehicle_engines_fuel_type', table_name='vehicle_engines')
    op.drop_table('vehicle_engines')
