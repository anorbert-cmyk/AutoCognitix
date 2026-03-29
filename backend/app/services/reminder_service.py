"""
Reminder notification service — sends email reminders for upcoming
maintenance reminders using the existing email_service.
"""

from datetime import date, datetime, timezone
from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.services.vehicle_garage_service import get_vehicle_garage_service

logger = get_logger(__name__)


class ReminderService:
    """Sends email notifications for upcoming maintenance reminders."""

    _instance: Optional["ReminderService"] = None
    _initialized: bool = False

    def __new__(cls) -> "ReminderService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        logger.info("ReminderService inicializálva")

    async def send_upcoming_reminders(
        self,
        db: AsyncSession,
        user_id: str,
        user_email: str,
        days_ahead: int = 7,
    ) -> int:
        """
        Send email for all unsent upcoming reminders within days_ahead.
        Returns number of emails sent.
        """
        from app.services.email_service import get_email_service

        garage_service = get_vehicle_garage_service()
        upcoming = await garage_service.get_upcoming_reminders(db, user_id, days_ahead)

        # Filter: only those not yet emailed
        to_notify = [r for r in upcoming if r.email_sent_at is None]

        if not to_notify:
            return 0

        email_service = get_email_service()
        sent_count = 0

        for reminder in to_notify:
            try:
                days_left: Optional[int] = None
                if reminder.due_date:
                    today = date.today()
                    days_left = (reminder.due_date - today).days

                subject = f"Emlékeztető: {reminder.title}"
                if days_left is not None:
                    if days_left < 0:
                        body = (
                            f"LEJÁRT emlékeztető: {reminder.title} ({abs(days_left)} napja lejárt)"
                        )
                    elif days_left == 0:
                        body = f"MA esedékes: {reminder.title}"
                    else:
                        body = (
                            f"{reminder.title} - {days_left} nap mulva esedékes "
                            f"({reminder.due_date})"
                        )
                else:
                    body = f"Emlékeztető: {reminder.title}"

                await email_service._send_email(
                    to_email=user_email,
                    subject=subject,
                    text_content=body,
                )

                # Mark as sent
                reminder.email_sent_at = datetime.now(timezone.utc)
                await db.flush()
                sent_count += 1
            except Exception as exc:
                logger.warning(f"Email küldés sikertelen: {reminder.id} - {exc}")

        return sent_count

    async def send_bulk_reminders(
        self,
        db: AsyncSession,
        users: List[dict],
        days_ahead: int = 7,
    ) -> int:
        """
        Send reminders for a list of users.
        Each item in users must have 'user_id' and 'email' keys.
        Returns total number of emails sent across all users.
        """
        total_sent = 0
        for user in users:
            user_id = user.get("user_id", "")
            user_email = user.get("email", "")
            if not user_id or not user_email:
                continue
            try:
                sent = await self.send_upcoming_reminders(
                    db=db,
                    user_id=user_id,
                    user_email=user_email,
                    days_ahead=days_ahead,
                )
                total_sent += sent
            except Exception as exc:
                logger.warning(f"Tömeges emlékeztető hiba ({user_id}): {exc}")
        return total_sent


_reminder_service_instance: Optional[ReminderService] = None


def get_reminder_service() -> ReminderService:
    """Get or create the singleton ReminderService instance."""
    global _reminder_service_instance
    if _reminder_service_instance is None:
        _reminder_service_instance = ReminderService()
    return _reminder_service_instance
