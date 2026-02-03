"""Initial schema for AutoCognitix database.

Revision ID: 001_initial_schema
Revises:
Create Date: 2024-01-01 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001_initial_schema'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create all tables."""

    # 1. users - felhasználók
    op.create_table(
        'users',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('email', sa.String(255), unique=True, index=True, nullable=False),
        sa.Column('hashed_password', sa.String(255), nullable=False),
        sa.Column('full_name', sa.String(100), nullable=True),
        sa.Column('is_active', sa.Boolean(), default=True, nullable=False),
        sa.Column('is_superuser', sa.Boolean(), default=False, nullable=False),
        sa.Column('role', sa.String(50), default='user', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
    )

    # 2. vehicle_makes - gyártók
    op.create_table(
        'vehicle_makes',
        sa.Column('id', sa.String(50), primary_key=True),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('country', sa.String(100), nullable=True),
        sa.Column('logo_url', sa.String(500), nullable=True),
    )

    # 3. vehicle_models - modellek
    op.create_table(
        'vehicle_models',
        sa.Column('id', sa.String(50), primary_key=True),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('make_id', sa.String(50), sa.ForeignKey('vehicle_makes.id'), nullable=False),
        sa.Column('year_start', sa.Integer(), nullable=False),
        sa.Column('year_end', sa.Integer(), nullable=True),
        sa.Column('body_types', postgresql.ARRAY(sa.String()), default=[], nullable=True),
        sa.Column('engine_codes', postgresql.ARRAY(sa.String()), default=[], nullable=True),
        sa.Column('platform', sa.String(50), nullable=True),
    )

    # Index for vehicle_models.make_id
    op.create_index('ix_vehicle_models_make_id', 'vehicle_models', ['make_id'])

    # 4. dtc_codes - hibakódok
    op.create_table(
        'dtc_codes',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('code', sa.String(10), unique=True, index=True, nullable=False),
        sa.Column('description_en', sa.Text(), nullable=False),
        sa.Column('description_hu', sa.Text(), nullable=True),
        sa.Column('category', sa.String(20), nullable=False),
        sa.Column('severity', sa.String(20), default='medium', nullable=False),
        sa.Column('is_generic', sa.Boolean(), default=True, nullable=False),
        sa.Column('system', sa.String(100), nullable=True),
        sa.Column('symptoms', postgresql.ARRAY(sa.String()), default=[], nullable=True),
        sa.Column('possible_causes', postgresql.ARRAY(sa.String()), default=[], nullable=True),
        sa.Column('diagnostic_steps', postgresql.ARRAY(sa.String()), default=[], nullable=True),
        sa.Column('related_codes', postgresql.ARRAY(sa.String()), default=[], nullable=True),
        sa.Column('manufacturer_code', sa.String(50), nullable=True),
        sa.Column('applicable_makes', postgresql.ARRAY(sa.String()), default=[], nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
    )

    # 5. known_issues - ismert hibák
    op.create_table(
        'known_issues',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('title', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('symptoms', postgresql.ARRAY(sa.String()), default=[], nullable=True),
        sa.Column('causes', postgresql.ARRAY(sa.String()), default=[], nullable=True),
        sa.Column('solutions', postgresql.ARRAY(sa.String()), default=[], nullable=True),
        sa.Column('related_dtc_codes', postgresql.ARRAY(sa.String()), default=[], nullable=True),
        sa.Column('applicable_makes', postgresql.ARRAY(sa.String()), default=[], nullable=True),
        sa.Column('applicable_models', postgresql.ARRAY(sa.String()), default=[], nullable=True),
        sa.Column('year_start', sa.Integer(), nullable=True),
        sa.Column('year_end', sa.Integer(), nullable=True),
        sa.Column('confidence', sa.Float(), default=0.5, nullable=False),
        sa.Column('source_type', sa.String(50), nullable=True),
        sa.Column('source_url', sa.String(500), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
    )

    # 6. diagnosis_sessions - diagnosztikai munkamenetek
    op.create_table(
        'diagnosis_sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id'), nullable=True),
        sa.Column('vehicle_make', sa.String(100), nullable=False),
        sa.Column('vehicle_model', sa.String(100), nullable=False),
        sa.Column('vehicle_year', sa.Integer(), nullable=False),
        sa.Column('vehicle_vin', sa.String(17), nullable=True),
        sa.Column('dtc_codes', postgresql.ARRAY(sa.String()), nullable=False),
        sa.Column('symptoms_text', sa.Text(), nullable=False),
        sa.Column('additional_context', sa.Text(), nullable=True),
        sa.Column('diagnosis_result', postgresql.JSONB(), nullable=False),
        sa.Column('confidence_score', sa.Float(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # Index for diagnosis_sessions.user_id
    op.create_index('ix_diagnosis_sessions_user_id', 'diagnosis_sessions', ['user_id'])


def downgrade() -> None:
    """Drop all tables in reverse order (FK dependencies)."""

    # Drop indexes first
    op.drop_index('ix_diagnosis_sessions_user_id', table_name='diagnosis_sessions')
    op.drop_index('ix_vehicle_models_make_id', table_name='vehicle_models')

    # Drop tables in reverse order (FK constraints)
    op.drop_table('diagnosis_sessions')
    op.drop_table('known_issues')
    op.drop_table('dtc_codes')
    op.drop_table('vehicle_models')
    op.drop_table('vehicle_makes')
    op.drop_table('users')
