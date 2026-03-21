"""Newsletter subscribers table.

Revision ID: 013_newsletter_subscribers
Revises: 012_epa_vehicles
Create Date: 2026-03-21
"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "013_newsletter_subscribers"
down_revision: Union[str, None] = "012_epa_vehicles"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "newsletter_subscribers",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("email", sa.String(255), unique=True, index=True, nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("confirm_token", sa.String(255), index=True),
        sa.Column("unsubscribe_token", sa.String(255), unique=True, nullable=False),
        sa.Column("source", sa.String(50), server_default="landing_page"),
        sa.Column("language", sa.String(5), server_default="hu"),
        sa.Column("ip_address", sa.String(45)),
        sa.Column("confirmed_at", sa.DateTime(timezone=True)),
        sa.Column("unsubscribed_at", sa.DateTime(timezone=True)),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )


def downgrade() -> None:
    op.drop_table("newsletter_subscribers")
