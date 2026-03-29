"""
Vehicle Garage Service — CRUD operations for user vehicles,
maintenance reminders, and maintenance cost tracking.
"""

from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.core.log_sanitizer import sanitize_log
from app.db.postgres.models import MaintenanceCost, MaintenanceReminder, UserVehicle

logger = get_logger(__name__)

REMINDER_TYPE_LABELS: Dict[str, str] = {
    "oil_change": "Olajcsere",
    "tire_rotation": "Gumicsere / Forgatás",
    "mueszaki_vizsga": "Műszaki vizsga",
    "kotelezo_biztositas": "Kötelező biztosítás megújítás",
    "coolant": "Hűtőfolyadék csere",
    "brake_fluid": "Fékfolyadék csere",
    "timing_belt": "Vezérszíj csere",
    "air_filter": "Légszűrő csere",
    "brake_pads": "Fékbetét csere",
    "custom": "Egyedi emlékeztető",
}


class VehicleGarageServiceError(Exception):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class VehicleGarageService:
    """Service for managing user vehicle garage, reminders, and maintenance costs."""

    _instance: Optional["VehicleGarageService"] = None
    _initialized: bool = False

    def __new__(cls) -> "VehicleGarageService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        logger.info("VehicleGarageService inicializálva")

    # ─── UserVehicle CRUD ──────────────────────────────────────────────────────

    async def create_vehicle(
        self,
        db: AsyncSession,
        user_id: str,
        data: Dict[str, Any],
    ) -> UserVehicle:
        """Create a new vehicle for a user."""
        try:
            vehicle = UserVehicle(
                id=str(uuid4()),
                user_id=user_id,
                **{k: v for k, v in data.items() if v is not None},
            )
            db.add(vehicle)
            await db.flush()
            await db.refresh(vehicle)
            logger.info(
                "Új jármű létrehozva",
                extra={"user_id": sanitize_log(user_id), "vehicle_id": vehicle.id},
            )
            return vehicle
        except Exception as exc:
            logger.error(f"Jármű létrehozás hiba: {exc}", exc_info=True)
            raise VehicleGarageServiceError("Nem sikerült létrehozni a járművet") from exc

    async def get_vehicles(
        self,
        db: AsyncSession,
        user_id: str,
        skip: int = 0,
        limit: int = 50,
    ) -> Tuple[List[UserVehicle], int]:
        """List all active vehicles for a user."""
        stmt = (
            select(UserVehicle)
            .where(and_(UserVehicle.user_id == user_id, UserVehicle.is_active.is_(True)))
            .order_by(UserVehicle.created_at.desc())
        )
        count_stmt = (
            select(func.count())
            .select_from(UserVehicle)
            .where(and_(UserVehicle.user_id == user_id, UserVehicle.is_active.is_(True)))
        )
        count_result = await db.execute(count_stmt)
        total = count_result.scalar() or 0
        result = await db.execute(stmt.offset(skip).limit(limit))
        vehicles = list(result.scalars().all())
        return vehicles, total

    async def get_vehicle(
        self,
        db: AsyncSession,
        vehicle_id: str,
        user_id: str,
    ) -> Optional[UserVehicle]:
        """Get a single vehicle by ID, verifying ownership."""
        result = await db.execute(
            select(UserVehicle).where(
                and_(UserVehicle.id == vehicle_id, UserVehicle.user_id == user_id)
            )
        )
        return result.scalar_one_or_none()  # type: ignore[no-any-return]

    async def update_vehicle(
        self,
        db: AsyncSession,
        vehicle_id: str,
        user_id: str,
        data: Dict[str, Any],
    ) -> Optional[UserVehicle]:
        """Update vehicle fields. Returns None if not found or not owned."""
        vehicle = await self.get_vehicle(db, vehicle_id, user_id)
        if not vehicle:
            return None
        for key, value in data.items():
            if value is not None:
                setattr(vehicle, key, value)
        vehicle.updated_at = datetime.now(timezone.utc)
        await db.flush()
        await db.refresh(vehicle)
        return vehicle

    async def delete_vehicle(
        self,
        db: AsyncSession,
        vehicle_id: str,
        user_id: str,
    ) -> bool:
        """Soft-delete a vehicle (set is_active=False)."""
        vehicle = await self.get_vehicle(db, vehicle_id, user_id)
        if not vehicle:
            return False
        vehicle.is_active = False
        vehicle.updated_at = datetime.now(timezone.utc)
        await db.flush()
        logger.info("Jármű törölve (soft)", extra={"vehicle_id": vehicle_id})
        return True

    # ─── Health Score ──────────────────────────────────────────────────────────

    async def get_health_score(
        self,
        db: AsyncSession,
        vehicle_id: str,
        user_id: str,
    ) -> Dict[str, Any]:
        """
        Calculate vehicle health score (0-100) based on:
        - Active overdue reminders (-10 each)
        - Completed reminders in last 6 months (+3 each, max +15)
        - Diagnosis sessions in last 90 days (+5, max +10)
        Base score: 80
        """
        score = 80
        factors: List[Dict[str, Any]] = []

        # Overdue reminders
        today = date.today()
        overdue_stmt = (
            select(func.count())
            .select_from(MaintenanceReminder)
            .where(
                and_(
                    MaintenanceReminder.vehicle_id == vehicle_id,
                    MaintenanceReminder.is_completed.is_(False),
                    MaintenanceReminder.due_date < today,
                )
            )
        )
        overdue_result = await db.execute(overdue_stmt)
        overdue_count = overdue_result.scalar() or 0
        if overdue_count > 0:
            penalty = min(overdue_count * 10, 40)
            score -= penalty
            factors.append(
                {
                    "type": "negative",
                    "label": f"{overdue_count} lejárt emlékeztető",
                    "impact": -penalty,
                }
            )

        # Completed reminders (last 6 months)
        six_months_ago = today - timedelta(days=180)
        completed_stmt = (
            select(func.count())
            .select_from(MaintenanceReminder)
            .where(
                and_(
                    MaintenanceReminder.vehicle_id == vehicle_id,
                    MaintenanceReminder.is_completed.is_(True),
                    MaintenanceReminder.completed_at
                    >= datetime(
                        six_months_ago.year,
                        six_months_ago.month,
                        six_months_ago.day,
                        tzinfo=timezone.utc,
                    ),
                )
            )
        )
        completed_result = await db.execute(completed_stmt)
        completed_count = completed_result.scalar() or 0
        if completed_count > 0:
            bonus = min(completed_count * 3, 15)
            score += bonus
            factors.append(
                {
                    "type": "positive",
                    "label": f"{completed_count} elvégzett karbantartás (6 hónap)",
                    "impact": bonus,
                }
            )

        score = max(0, min(100, score))

        if score >= 80:
            category = "kiváló"
            color = "green"
        elif score >= 60:
            category = "jó"
            color = "yellow"
        elif score >= 40:
            category = "figyelmet_igényel"
            color = "orange"
        else:
            category = "kritikus"
            color = "red"

        return {
            "vehicle_id": vehicle_id,
            "score": score,
            "category": category,
            "category_color": color,
            "factors": factors,
        }

    # ─── MaintenanceReminder CRUD ──────────────────────────────────────────────

    async def create_reminder(
        self,
        db: AsyncSession,
        user_id: str,
        data: Dict[str, Any],
    ) -> MaintenanceReminder:
        """Create a new maintenance reminder."""
        try:
            reminder = MaintenanceReminder(id=str(uuid4()), user_id=user_id, **data)
            db.add(reminder)
            await db.flush()
            await db.refresh(reminder)
            return reminder
        except Exception as exc:
            logger.error(f"Emlékeztető létrehozás hiba: {exc}", exc_info=True)
            raise VehicleGarageServiceError("Nem sikerült létrehozni az emlékeztetőt") from exc

    async def get_reminders(
        self,
        db: AsyncSession,
        user_id: str,
        vehicle_id: Optional[str] = None,
        include_completed: bool = False,
    ) -> Tuple[List[MaintenanceReminder], int]:
        """List reminders for user, optionally filtered by vehicle."""
        conditions = [MaintenanceReminder.user_id == user_id]
        if vehicle_id:
            conditions.append(MaintenanceReminder.vehicle_id == vehicle_id)
        if not include_completed:
            conditions.append(MaintenanceReminder.is_completed.is_(False))

        stmt = (
            select(MaintenanceReminder)
            .where(and_(*conditions))
            .order_by(MaintenanceReminder.due_date.asc().nulls_last())
        )
        count_stmt = select(func.count()).select_from(MaintenanceReminder).where(and_(*conditions))
        count_result = await db.execute(count_stmt)
        total = count_result.scalar() or 0
        result = await db.execute(stmt)
        reminders = list(result.scalars().all())
        return reminders, total

    async def get_upcoming_reminders(
        self,
        db: AsyncSession,
        user_id: str,
        days_ahead: int = 30,
    ) -> List[MaintenanceReminder]:
        """Get reminders due within the next N days (including overdue)."""
        today = date.today()
        cutoff = today + timedelta(days=days_ahead)
        stmt = (
            select(MaintenanceReminder)
            .where(
                and_(
                    MaintenanceReminder.user_id == user_id,
                    MaintenanceReminder.is_completed.is_(False),
                    MaintenanceReminder.due_date <= cutoff,
                )
            )
            .order_by(MaintenanceReminder.due_date.asc().nulls_last())
            .limit(20)
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def complete_reminder(
        self,
        db: AsyncSession,
        reminder_id: str,
        user_id: str,
    ) -> Optional[MaintenanceReminder]:
        """Mark a reminder as completed."""
        result = await db.execute(
            select(MaintenanceReminder).where(
                and_(
                    MaintenanceReminder.id == reminder_id,
                    MaintenanceReminder.user_id == user_id,
                )
            )
        )
        reminder = result.scalar_one_or_none()
        if not reminder:
            return None
        reminder.is_completed = True
        reminder.completed_at = datetime.now(timezone.utc)
        reminder.updated_at = datetime.now(timezone.utc)
        await db.flush()
        await db.refresh(reminder)
        return reminder  # type: ignore[no-any-return]

    async def delete_reminder(
        self,
        db: AsyncSession,
        reminder_id: str,
        user_id: str,
    ) -> bool:
        """Delete a reminder (hard delete)."""
        result = await db.execute(
            select(MaintenanceReminder).where(
                and_(
                    MaintenanceReminder.id == reminder_id,
                    MaintenanceReminder.user_id == user_id,
                )
            )
        )
        reminder = result.scalar_one_or_none()
        if not reminder:
            return False
        await db.delete(reminder)
        await db.flush()
        return True

    # ─── MaintenanceCost CRUD ──────────────────────────────────────────────────

    async def create_cost(
        self,
        db: AsyncSession,
        user_id: str,
        data: Dict[str, Any],
    ) -> MaintenanceCost:
        """Record a maintenance cost entry."""
        try:
            cost = MaintenanceCost(id=str(uuid4()), user_id=user_id, **data)
            db.add(cost)
            await db.flush()
            await db.refresh(cost)
            return cost
        except Exception as exc:
            logger.error(f"Karbantartási költség rögzítés hiba: {exc}", exc_info=True)
            raise VehicleGarageServiceError("Nem sikerült rögzíteni a költséget") from exc

    async def get_costs(
        self,
        db: AsyncSession,
        user_id: str,
        vehicle_id: Optional[str] = None,
        limit: int = 50,
    ) -> Tuple[List[MaintenanceCost], int, int]:
        """
        Get maintenance costs.
        Returns: (costs, total_count, total_cost_huf)
        """
        conditions = [MaintenanceCost.user_id == user_id]
        if vehicle_id:
            conditions.append(MaintenanceCost.vehicle_id == vehicle_id)

        stmt = (
            select(MaintenanceCost)
            .where(and_(*conditions))
            .order_by(MaintenanceCost.service_date.desc())
            .limit(limit)
        )
        count_stmt = select(func.count()).select_from(MaintenanceCost).where(and_(*conditions))
        sum_stmt = select(func.sum(MaintenanceCost.cost_huf)).where(and_(*conditions))

        count_result = await db.execute(count_stmt)
        sum_result = await db.execute(sum_stmt)
        result = await db.execute(stmt)

        total = count_result.scalar() or 0
        total_cost = sum_result.scalar() or 0
        costs = list(result.scalars().all())
        return costs, total, total_cost


# ─── Factory ──────────────────────────────────────────────────────────────────

_garage_service_instance: Optional[VehicleGarageService] = None


def get_vehicle_garage_service() -> VehicleGarageService:
    """Get or create the singleton VehicleGarageService instance."""
    global _garage_service_instance
    if _garage_service_instance is None:
        _garage_service_instance = VehicleGarageService()
    return _garage_service_instance
