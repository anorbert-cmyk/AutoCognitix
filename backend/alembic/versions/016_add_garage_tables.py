"""Add garage tables: user_vehicles, maintenance_reminders, maintenance_costs.

Revision ID: 016_add_garage_tables
Revises: 015_merge_heads
Create Date: 2026-03-29
"""

from typing import Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "016_add_garage_tables"  # lgtm[py/unused-global-variable]
down_revision: Union[str, None] = "015_merge_heads"  # lgtm[py/unused-global-variable]


def upgrade() -> None:
    op.create_table(
        "user_vehicles",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("nickname", sa.String(100)),
        sa.Column("make", sa.String(100), nullable=False),
        sa.Column("model", sa.String(100), nullable=False),
        sa.Column("year", sa.Integer, nullable=False),
        sa.Column("vin", sa.String(17)),
        sa.Column("license_plate", sa.String(20)),
        sa.Column("mileage_km", sa.Integer),
        sa.Column("fuel_type", sa.String(30)),
        sa.Column("color", sa.String(50)),
        sa.Column("notes", sa.Text),
        sa.Column("is_active", sa.Boolean, default=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_user_vehicles_user_id", "user_vehicles", ["user_id"])
    op.create_index("ix_user_vehicles_make", "user_vehicles", ["make"])
    op.create_index("ix_user_vehicles_model", "user_vehicles", ["model"])
    op.create_index("ix_user_vehicles_vin", "user_vehicles", ["vin"])
    op.create_index("ix_user_vehicles_license_plate", "user_vehicles", ["license_plate"])

    op.create_table(
        "maintenance_reminders",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "vehicle_id",
            UUID(as_uuid=True),
            sa.ForeignKey("user_vehicles.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "user_id",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("reminder_type", sa.String(50), nullable=False),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("due_date", sa.Date),
        sa.Column("due_mileage_km", sa.Integer),
        sa.Column("notes", sa.Text),
        sa.Column("is_completed", sa.Boolean, default=False, nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("email_sent_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_maintenance_reminders_vehicle_id", "maintenance_reminders", ["vehicle_id"])
    op.create_index("ix_maintenance_reminders_user_id", "maintenance_reminders", ["user_id"])
    op.create_index("ix_maintenance_reminders_due_date", "maintenance_reminders", ["due_date"])
    op.create_index(
        "ix_maintenance_reminders_is_completed", "maintenance_reminders", ["is_completed"]
    )
    op.create_index(
        "ix_maintenance_reminders_reminder_type", "maintenance_reminders", ["reminder_type"]
    )

    op.create_table(
        "maintenance_costs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "vehicle_id",
            UUID(as_uuid=True),
            sa.ForeignKey("user_vehicles.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "user_id",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "diagnosis_session_id",
            UUID(as_uuid=True),
            sa.ForeignKey("diagnosis_sessions.id", ondelete="SET NULL"),
        ),
        sa.Column("service_type", sa.String(100), nullable=False),
        sa.Column("cost_huf", sa.Integer, nullable=False),
        sa.Column("service_date", sa.Date, nullable=False),
        sa.Column("mileage_km", sa.Integer),
        sa.Column("workshop_name", sa.String(200)),
        sa.Column("notes", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_maintenance_costs_vehicle_id", "maintenance_costs", ["vehicle_id"])
    op.create_index("ix_maintenance_costs_user_id", "maintenance_costs", ["user_id"])
    op.create_index("ix_maintenance_costs_service_date", "maintenance_costs", ["service_date"])
    op.create_index(
        "ix_maintenance_costs_diagnosis_session_id",
        "maintenance_costs",
        ["diagnosis_session_id"],
    )


def downgrade() -> None:
    # Drop indexes explicitly before tables (best practice for Alembic downgrade symmetry)
    op.drop_index("ix_maintenance_costs_diagnosis_session_id", table_name="maintenance_costs")
    op.drop_index("ix_maintenance_costs_service_date", table_name="maintenance_costs")
    op.drop_index("ix_maintenance_costs_user_id", table_name="maintenance_costs")
    op.drop_index("ix_maintenance_costs_vehicle_id", table_name="maintenance_costs")
    op.drop_table("maintenance_costs")

    op.drop_index("ix_maintenance_reminders_reminder_type", table_name="maintenance_reminders")
    op.drop_index("ix_maintenance_reminders_is_completed", table_name="maintenance_reminders")
    op.drop_index("ix_maintenance_reminders_due_date", table_name="maintenance_reminders")
    op.drop_index("ix_maintenance_reminders_user_id", table_name="maintenance_reminders")
    op.drop_index("ix_maintenance_reminders_vehicle_id", table_name="maintenance_reminders")
    op.drop_table("maintenance_reminders")

    op.drop_index("ix_user_vehicles_license_plate", table_name="user_vehicles")
    op.drop_index("ix_user_vehicles_vin", table_name="user_vehicles")
    op.drop_index("ix_user_vehicles_model", table_name="user_vehicles")
    op.drop_index("ix_user_vehicles_make", table_name="user_vehicles")
    op.drop_index("ix_user_vehicles_user_id", table_name="user_vehicles")
    op.drop_table("user_vehicles")
