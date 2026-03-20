"""
Diagnosis archival service.

Archives old/deleted diagnosis sessions to JSONB storage for
space efficiency while preserving data for compliance (GDPR).
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# Archive sessions older than 90 days
DEFAULT_ARCHIVE_AGE_DAYS = 90


class ArchiveService:
    """Handles archival of old diagnosis sessions."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def archive_old_sessions(
        self,
        age_days: int = DEFAULT_ARCHIVE_AGE_DAYS,
        batch_size: int = 100,
    ) -> int:
        """
        Archive diagnosis sessions older than age_days.

        Uses SELECT FOR UPDATE to prevent concurrent archival of same sessions.
        Each session is archived atomically via savepoint.

        Returns the number of sessions archived.
        """
        from app.db.postgres.models import DiagnosisArchive, DiagnosisSession

        cutoff = datetime.now(timezone.utc) - timedelta(days=age_days)

        # Lock rows to prevent concurrent archival race condition
        result = await self.db.execute(
            select(DiagnosisSession)
            .where(
                and_(
                    DiagnosisSession.created_at < cutoff,
                    DiagnosisSession.is_deleted.is_(False),
                )
            )
            .with_for_update(skip_locked=True)
            .limit(batch_size)
        )
        sessions = result.scalars().all()

        if not sessions:
            return 0

        archived_count = 0
        for session in sessions:
            # Use savepoint so one failure doesn't roll back entire batch
            savepoint = await self.db.begin_nested()
            try:
                archive = DiagnosisArchive(
                    id=uuid4(),
                    original_id=session.id,
                    user_id=session.user_id,
                    original_created_at=session.created_at,
                    session_data={
                        "dtc_codes": session.dtc_codes,
                        "symptoms_text": session.symptoms_text,
                        "additional_context": session.additional_context,
                        "diagnosis_result": session.diagnosis_result,
                        "confidence_score": session.confidence_score,
                        "vehicle_make": session.vehicle_make,
                        "vehicle_model": session.vehicle_model,
                        "vehicle_year": session.vehicle_year,
                        "vehicle_vin": session.vehicle_vin,
                    },
                    dtc_codes=session.dtc_codes,
                    vehicle_info={
                        "make": session.vehicle_make,
                        "model": session.vehicle_model,
                        "year": session.vehicle_year,
                        "vin": session.vehicle_vin,
                    },
                )
                self.db.add(archive)

                # Soft delete the original
                session.is_deleted = True
                session.deleted_at = datetime.now(timezone.utc)

                await savepoint.commit()
                archived_count += 1
            except Exception:
                await savepoint.rollback()
                logger.exception("Failed to archive session %s", session.id)
                continue

        logger.info("Archived %d/%d diagnosis sessions", archived_count, len(sessions))
        return archived_count

    async def get_archived_session(self, original_id: UUID) -> Optional[dict]:
        """Retrieve an archived session by its original ID."""
        from app.db.postgres.models import DiagnosisArchive

        result = await self.db.execute(
            select(DiagnosisArchive).where(DiagnosisArchive.original_id == original_id)
        )
        archive = result.scalar_one_or_none()
        if archive:
            return archive.session_data
        return None
