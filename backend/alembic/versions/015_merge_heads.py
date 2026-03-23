"""merge newsletter and diagnosis archive branches

Revision ID: 0c29b63434c4
Revises: 013_newsletter_subscribers, 014_add_diagnosis_dedup_index
Create Date: 2026-03-23 08:26:40.409234

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '015_merge_heads'
down_revision: Union[str, None] = ('013_newsletter_subscribers', '014_add_diagnosis_dedup_index')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
