"""Add security columns to users table.

Revision ID: 011_add_user_security_columns
Revises: 010_reseed_vehicle_data
Create Date: 2026-02-09 15:30:00.000000

Adds account lockout, password reset, email verification, and login tracking
columns that exist in the SQLAlchemy model but were missing from migrations.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "011_add_user_security_columns"
down_revision: str = "010_reseed_vehicle_data"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Account lockout fields
    op.add_column(
        "users",
        sa.Column("failed_login_attempts", sa.Integer(), server_default="0", nullable=False),
    )
    op.add_column("users", sa.Column("locked_until", sa.DateTime(timezone=True), nullable=True))
    op.add_column(
        "users", sa.Column("last_failed_login", sa.DateTime(timezone=True), nullable=True)
    )

    # Password reset fields
    op.add_column("users", sa.Column("password_reset_token", sa.String(255), nullable=True))
    op.add_column(
        "users", sa.Column("password_reset_expires", sa.DateTime(timezone=True), nullable=True)
    )

    # Email verification fields
    op.add_column(
        "users",
        sa.Column("is_email_verified", sa.Boolean(), server_default="false", nullable=False),
    )
    op.add_column("users", sa.Column("email_verification_token", sa.String(255), nullable=True))

    # Login tracking
    op.add_column("users", sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "last_login_at")
    op.drop_column("users", "email_verification_token")
    op.drop_column("users", "is_email_verified")
    op.drop_column("users", "password_reset_expires")
    op.drop_column("users", "password_reset_token")
    op.drop_column("users", "last_failed_login")
    op.drop_column("users", "locked_until")
    op.drop_column("users", "failed_login_attempts")
