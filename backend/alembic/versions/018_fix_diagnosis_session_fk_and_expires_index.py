"""fix DiagnosisSession FK ondelete and PasswordResetToken expires_at index

Revision ID: 018_fix_diagnosis_fk
Revises: 017_add_password_reset_tokens
Create Date: 2026-03-29
"""  # lgtm[py/unused-global-variable]

from typing import Union

from alembic import op

revision: str = "018_fix_diagnosis_fk"  # lgtm[py/unused-global-variable]
down_revision: Union[str, None] = "017_add_password_reset_tokens"  # lgtm[py/unused-global-variable]


def upgrade() -> None:
    # DiagnosisSession FK fix: add ondelete="SET NULL" (nullable field)
    op.drop_constraint("diagnosis_sessions_user_id_fkey", "diagnosis_sessions", type_="foreignkey")
    op.create_foreign_key(
        "diagnosis_sessions_user_id_fkey",
        "diagnosis_sessions",
        "users",
        ["user_id"],
        ["id"],
        ondelete="SET NULL",
    )

    # PasswordResetToken expires_at index for efficient cleanup queries
    op.create_index(
        "ix_password_reset_tokens_expires_at",
        "password_reset_tokens",
        ["expires_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_password_reset_tokens_expires_at", table_name="password_reset_tokens")
    op.drop_constraint("diagnosis_sessions_user_id_fkey", "diagnosis_sessions", type_="foreignkey")
    op.create_foreign_key(
        "diagnosis_sessions_user_id_fkey",
        "diagnosis_sessions",
        "users",
        ["user_id"],
        ["id"],
    )
