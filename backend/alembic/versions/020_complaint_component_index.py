"""add functional index for the NHTSA complaint-component aggregation

The `/vehicles/{make}/{model}/common-issues` endpoint now ranks a vehicle's
components by NHTSA complaint frequency with a grouped aggregation over
`vehicle_complaints` (~751K rows). Its predicate is::

    lower(make) IN (...) AND lower(model) LIKE 'golf%' [AND model_year = ...]

Migration 003 already created `ix_vehicle_complaints_make_model_year` on the
RAW `(make, model, model_year)` columns, but that index is unusable here:

1. The stored values are uppercase (`VOLKSWAGEN`, `GOLF GTI`) while callers
   pass user spellings, so the query wraps both columns in `lower()`. A plain
   btree on `make`/`model` cannot serve `lower(make) = ...`.
2. Under a non-C collation (Railway PostgreSQL defaults to `en_US.utf8`) a
   btree with the default `text_ops` opclass cannot serve a `LIKE 'prefix%'`
   range scan either - that needs `text_pattern_ops`.

Without this index the aggregation is a full sequential scan of every complaint
row on each request. With it, `lower(make)` gives the leading equality and
`lower(model) text_pattern_ops` gives the prefix range; `model_year` rides along
as an in-index filter so the optional year filter does not push the scan back to
the heap.

Not created CONCURRENTLY: Alembic runs migrations inside a transaction and
CREATE INDEX CONCURRENTLY cannot run in one. `vehicle_complaints` is a
bulk-loaded reference table refreshed on a quarterly NHTSA cadence, never on the
request path, so the brief ACCESS EXCLUSIVE lock (seconds at this row count) is
acceptable. If the table grows by an order of magnitude, split this into an
out-of-band CONCURRENTLY build instead.

Revision ID: 020_complaint_component_idx
Revises: 019_fix_archive_drift
Create Date: 2026-07-25
"""

from typing import Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic at runtime via globals() introspection.
# The static analyzer can't see that usage, so we mark them explicitly with the
# `lgtm[...]` directive this project uses everywhere else (CLAUDE.md; migrations
# 016/017/018). A `codeql[...]` comment is not a suppression the scanner honours,
# so these two lines would still have been reported.
revision: str = "020_complaint_component_idx"  # lgtm[py/unused-global-variable]
down_revision: Union[str, None] = "019_fix_archive_drift"  # lgtm[py/unused-global-variable]

__all__ = ["revision", "down_revision", "upgrade", "downgrade"]

INDEX_NAME = "ix_vehicle_complaints_lower_make_model"


def upgrade() -> None:
    # Raw SQL rather than op.create_index: the latter has no way to attach a
    # per-column opclass (`text_pattern_ops`), which is the entire point here.
    op.execute(
        sa.text(
            f"CREATE INDEX IF NOT EXISTS {INDEX_NAME} "
            "ON vehicle_complaints "
            "(lower(make), lower(model) text_pattern_ops, model_year)"
        )
    )


def downgrade() -> None:
    # Explicitly drops exactly what upgrade() created - nothing else. The
    # migration-003 indexes on the raw columns are untouched by both directions.
    op.execute(sa.text(f"DROP INDEX IF EXISTS {INDEX_NAME}"))
