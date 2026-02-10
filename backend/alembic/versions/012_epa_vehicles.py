"""EPA FuelEconomy.gov vehicles table.

Revision ID: 012_epa_vehicles
Revises: 011_add_user_security_columns
Create Date: 2026-02-10
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "012_epa_vehicles"
down_revision: str = "011_add_user_security_columns"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create epa_vehicles table with EPA FuelEconomy.gov vehicle specifications."""
    op.create_table(
        "epa_vehicles",
        sa.Column("id", sa.Integer(), nullable=False),  # EPA source id
        sa.Column("make", sa.String(100), nullable=False),
        sa.Column("model", sa.String(200), nullable=False),
        sa.Column("base_model", sa.String(200), nullable=True),
        sa.Column("model_year", sa.Integer(), nullable=False),
        sa.Column("vehicle_class", sa.String(100), nullable=True),
        sa.Column("drive_type", sa.String(100), nullable=True),
        sa.Column("transmission", sa.String(100), nullable=True),
        sa.Column("cylinders", sa.Integer(), nullable=True),
        sa.Column("displacement_liters", sa.Float(), nullable=True),
        sa.Column("engine_description", sa.String(300), nullable=True),
        sa.Column("fuel_type", sa.String(50), nullable=True),
        sa.Column("fuel_category", sa.String(100), nullable=True),
        sa.Column("mpg_city", sa.Integer(), nullable=True),
        sa.Column("mpg_highway", sa.Integer(), nullable=True),
        sa.Column("mpg_combined", sa.Integer(), nullable=True),
        sa.Column("co2_grams_per_mile", sa.Float(), nullable=True),
        sa.Column("ev_motor", sa.String(200), nullable=True),
        sa.Column("range_miles", sa.Integer(), nullable=True),
        sa.Column("has_turbo", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("has_supercharger", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    # Individual indexes
    op.create_index("ix_epa_vehicles_make", "epa_vehicles", ["make"])
    op.create_index("ix_epa_vehicles_model", "epa_vehicles", ["model"])
    op.create_index("ix_epa_vehicles_model_year", "epa_vehicles", ["model_year"])
    op.create_index("ix_epa_vehicles_fuel_type", "epa_vehicles", ["fuel_type"])
    # Composite index for fast lookups
    op.create_index(
        "ix_epa_vehicles_make_model_year", "epa_vehicles", ["make", "model", "model_year"]
    )


def downgrade() -> None:
    """Drop epa_vehicles table and all indexes."""
    op.drop_index("ix_epa_vehicles_make_model_year", table_name="epa_vehicles")
    op.drop_index("ix_epa_vehicles_fuel_type", table_name="epa_vehicles")
    op.drop_index("ix_epa_vehicles_model_year", table_name="epa_vehicles")
    op.drop_index("ix_epa_vehicles_model", table_name="epa_vehicles")
    op.drop_index("ix_epa_vehicles_make", table_name="epa_vehicles")
    op.drop_table("epa_vehicles")
