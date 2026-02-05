"""Add performance indexes for query optimization.

Revision ID: 004_add_performance_indexes
Revises: 003_add_vehicle_recalls_complaints
Create Date: 2026-02-05 00:00:00.000000

Performance optimizations:
- Add composite indexes for frequently used query patterns
- Add partial indexes for filtered queries
- Add GIN indexes for ARRAY columns (contains queries)
- Add full-text search indexes for description columns
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '004_add_performance_indexes'
down_revision: Union[str, None] = '003_add_vehicle_recalls_complaints'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add performance indexes."""

    # =========================================================================
    # DTC Codes Table Indexes
    # =========================================================================

    # Composite index for category + severity (common filter combination)
    op.create_index(
        'ix_dtc_codes_category_severity',
        'dtc_codes',
        ['category', 'severity'],
        postgresql_using='btree'
    )

    # Partial index for generic codes only (most common query pattern)
    op.execute("""
        CREATE INDEX ix_dtc_codes_generic_codes
        ON dtc_codes (code)
        WHERE is_generic = true
    """)

    # GIN index for related_codes ARRAY column (contains queries)
    op.execute("""
        CREATE INDEX ix_dtc_codes_related_codes_gin
        ON dtc_codes USING GIN (related_codes)
    """)

    # GIN index for symptoms ARRAY (for symptom-based searches)
    op.execute("""
        CREATE INDEX ix_dtc_codes_symptoms_gin
        ON dtc_codes USING GIN (symptoms)
    """)

    # GIN index for applicable_makes (manufacturer-specific queries)
    op.execute("""
        CREATE INDEX ix_dtc_codes_applicable_makes_gin
        ON dtc_codes USING GIN (applicable_makes)
    """)

    # Full-text search index for description columns (Hungarian + English)
    op.execute("""
        CREATE INDEX ix_dtc_codes_description_fts
        ON dtc_codes USING GIN (
            to_tsvector('simple', COALESCE(description_en, '') || ' ' || COALESCE(description_hu, ''))
        )
    """)

    # =========================================================================
    # Known Issues Table Indexes
    # =========================================================================

    # GIN index for related_dtc_codes ARRAY
    op.execute("""
        CREATE INDEX ix_known_issues_related_dtc_codes_gin
        ON known_issues USING GIN (related_dtc_codes)
    """)

    # GIN index for applicable_makes ARRAY
    op.execute("""
        CREATE INDEX ix_known_issues_applicable_makes_gin
        ON known_issues USING GIN (applicable_makes)
    """)

    # Composite index for vehicle filtering
    op.create_index(
        'ix_known_issues_year_range',
        'known_issues',
        ['year_start', 'year_end'],
        postgresql_using='btree'
    )

    # Partial index for high-confidence issues
    op.execute("""
        CREATE INDEX ix_known_issues_high_confidence
        ON known_issues (confidence DESC)
        WHERE confidence >= 0.7
    """)

    # =========================================================================
    # Diagnosis Sessions Table Indexes
    # =========================================================================

    # Composite index for user history queries (user_id + created_at DESC)
    op.create_index(
        'ix_diagnosis_sessions_user_history',
        'diagnosis_sessions',
        ['user_id', sa.text('created_at DESC')],
        postgresql_using='btree'
    )

    # Index for vehicle-based queries
    op.create_index(
        'ix_diagnosis_sessions_vehicle',
        'diagnosis_sessions',
        ['vehicle_make', 'vehicle_model', 'vehicle_year'],
        postgresql_using='btree'
    )

    # GIN index for DTC codes array in sessions
    op.execute("""
        CREATE INDEX ix_diagnosis_sessions_dtc_codes_gin
        ON diagnosis_sessions USING GIN (dtc_codes)
    """)

    # Partial index for recent sessions (last 30 days) - most commonly queried
    op.execute("""
        CREATE INDEX ix_diagnosis_sessions_recent
        ON diagnosis_sessions (created_at DESC)
        WHERE created_at > NOW() - INTERVAL '30 days'
    """)

    # =========================================================================
    # Vehicle Models Table Indexes
    # =========================================================================

    # Composite index for make_id + year range queries
    op.create_index(
        'ix_vehicle_models_make_year',
        'vehicle_models',
        ['make_id', 'year_start', 'year_end'],
        postgresql_using='btree'
    )

    # =========================================================================
    # Users Table Indexes
    # =========================================================================

    # Partial index for active users only
    op.execute("""
        CREATE INDEX ix_users_active
        ON users (email)
        WHERE is_active = true
    """)


def downgrade() -> None:
    """Remove performance indexes."""

    # DTC Codes
    op.drop_index('ix_dtc_codes_category_severity', table_name='dtc_codes')
    op.execute('DROP INDEX IF EXISTS ix_dtc_codes_generic_codes')
    op.execute('DROP INDEX IF EXISTS ix_dtc_codes_related_codes_gin')
    op.execute('DROP INDEX IF EXISTS ix_dtc_codes_symptoms_gin')
    op.execute('DROP INDEX IF EXISTS ix_dtc_codes_applicable_makes_gin')
    op.execute('DROP INDEX IF EXISTS ix_dtc_codes_description_fts')

    # Known Issues
    op.execute('DROP INDEX IF EXISTS ix_known_issues_related_dtc_codes_gin')
    op.execute('DROP INDEX IF EXISTS ix_known_issues_applicable_makes_gin')
    op.drop_index('ix_known_issues_year_range', table_name='known_issues')
    op.execute('DROP INDEX IF EXISTS ix_known_issues_high_confidence')

    # Diagnosis Sessions
    op.drop_index('ix_diagnosis_sessions_user_history', table_name='diagnosis_sessions')
    op.drop_index('ix_diagnosis_sessions_vehicle', table_name='diagnosis_sessions')
    op.execute('DROP INDEX IF EXISTS ix_diagnosis_sessions_dtc_codes_gin')
    op.execute('DROP INDEX IF EXISTS ix_diagnosis_sessions_recent')

    # Vehicle Models
    op.drop_index('ix_vehicle_models_make_year', table_name='vehicle_models')

    # Users
    op.execute('DROP INDEX IF EXISTS ix_users_active')
