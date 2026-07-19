"""
Vehicle Garage Service — CRUD operations for user vehicles,
maintenance reminders, and maintenance cost tracking.
"""

from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID, uuid4

from sqlalchemy import and_, case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.core.log_sanitizer import sanitize_log
from app.db.postgres.models import MaintenanceCost, MaintenanceReminder, UserVehicle

logger = get_logger(__name__)


class VehicleGarageServiceError(Exception):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


# ─── Health-score helpers (pure, unit-testable) ─────────────────────────────────


def _as_uuid(value: Union[str, UUID]) -> UUID:
    """Coerce a str/UUID identifier to a ``uuid.UUID``.

    Uuid columns bind through ``value.hex`` under the SQLite test harness, which
    requires a real ``uuid.UUID`` (asyncpg accepts either str or UUID). Mirrors the
    ``UUID(user_id)`` coercion used in the auth endpoints so garage id-queries are
    portable across PostgreSQL and SQLite.
    """
    return value if isinstance(value, UUID) else UUID(str(value))


def compute_health_score(overdue: int, completed_6mo: int) -> int:
    """Health score (0-100): base 80, -10 per overdue (cap -40), +3 per completed (cap +15)."""
    score = 80
    if overdue > 0:
        score -= min(overdue * 10, 40)
    if completed_6mo > 0:
        score += min(completed_6mo * 3, 15)
    return max(0, min(100, score))


def build_health_factors(overdue: int, completed_6mo: int) -> List[Dict[str, Any]]:
    """Human-readable factors behind a health score (penalties first, then bonuses)."""
    factors: List[Dict[str, Any]] = []
    if overdue > 0:
        factors.append(
            {
                "type": "negative",
                "label": f"{overdue} lejárt emlékeztető",
                "impact": -min(overdue * 10, 40),
            }
        )
    if completed_6mo > 0:
        factors.append(
            {
                "type": "positive",
                "label": f"{completed_6mo} elvégzett karbantartás (6 hónap)",
                "impact": min(completed_6mo * 3, 15),
            }
        )
    return factors


def categorize_health(score: int) -> Tuple[str, str]:
    """Map a score to its (category, category_color) label pair."""
    if score >= 80:
        return "kiváló", "green"
    if score >= 60:
        return "jó", "yellow"
    if score >= 40:
        return "figyelmet_igényel", "orange"
    return "kritikus", "red"


class VehicleGarageService:
    """Service for managing user vehicle garage, reminders, and maintenance costs."""

    def __init__(self) -> None:
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
            # UUID binds (not str) so the ORM stays portable across PostgreSQL and the
            # SQLite test harness, whose Uuid bind processor requires a real uuid.UUID.
            vehicle = UserVehicle(
                id=uuid4(),
                user_id=_as_uuid(user_id),
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
        uid = _as_uuid(user_id)
        stmt = (
            select(UserVehicle)
            .where(and_(UserVehicle.user_id == uid, UserVehicle.is_active.is_(True)))
            .order_by(UserVehicle.created_at.desc())
        )
        count_stmt = (
            select(func.count())
            .select_from(UserVehicle)
            .where(and_(UserVehicle.user_id == uid, UserVehicle.is_active.is_(True)))
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
                and_(
                    UserVehicle.id == _as_uuid(vehicle_id),
                    UserVehicle.user_id == _as_uuid(user_id),
                )
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
        logger.info("Jármű törölve (soft)", extra={"vehicle_id": sanitize_log(vehicle_id)})
        return True

    # ─── Health Score ──────────────────────────────────────────────────────────

    async def get_reminder_aggregates(
        self,
        db: AsyncSession,
        vehicle_ids: List[str],
        today: date,
        user_id: str,
        upcoming_window_days: int = 30,
    ) -> Dict[str, Dict[str, int]]:
        """Aggregate reminder counts per vehicle in a single grouped query.

        Returns ``{vehicle_id: {"overdue", "completed_6mo", "upcoming"}}`` keyed by
        stringified vehicle id. Vehicles with no reminders are absent from the map.
        Overdue reminders are also counted as upcoming (due date on/before the window).
        Scoped to ``user_id`` as defense-in-depth against cross-tenant reminder leakage.
        """
        if not vehicle_ids:
            return {}

        six_months_ago = today - timedelta(days=180)
        six_dt = datetime(
            six_months_ago.year,
            six_months_ago.month,
            six_months_ago.day,
            tzinfo=timezone.utc,
        )
        upcoming_cutoff = today + timedelta(days=upcoming_window_days)

        rem = MaintenanceReminder

        def _agg(condition):
            return func.coalesce(func.sum(case((condition, 1), else_=0)), 0)

        overdue = and_(
            rem.is_completed.is_(False),
            rem.due_date.is_not(None),
            rem.due_date < today,
        )
        done6 = and_(
            rem.is_completed.is_(True),
            rem.completed_at.is_not(None),
            rem.completed_at >= six_dt,
        )
        upcoming = and_(
            rem.is_completed.is_(False),
            rem.due_date.is_not(None),
            rem.due_date <= upcoming_cutoff,
        )

        stmt = (
            select(
                rem.vehicle_id,
                _agg(overdue).label("o"),
                _agg(done6).label("c"),
                _agg(upcoming).label("u"),
            )
            .where(
                and_(
                    rem.vehicle_id.in_([_as_uuid(v) for v in vehicle_ids]),
                    rem.user_id == _as_uuid(user_id),
                )
            )
            .group_by(rem.vehicle_id)
        )
        result = await db.execute(stmt)
        return {
            str(row.vehicle_id): {
                "overdue": int(row.o or 0),
                "completed_6mo": int(row.c or 0),
                "upcoming": int(row.u or 0),
            }
            for row in result.all()
        }

    async def get_health_score(
        self,
        db: AsyncSession,
        vehicle_id: str,
        user_id: str,
    ) -> Dict[str, Any]:
        """
        Calculate vehicle health score (0-100) based on:
        - Active overdue reminders (-10 each, max -40)
        - Completed reminders in last 6 months (+3 each, max +15)
        Base score: 80
        """
        # Ownership check — prevent IDOR
        vehicle = await self.get_vehicle(db, vehicle_id, user_id)
        if not vehicle:
            raise VehicleGarageServiceError(
                f"Vehicle {sanitize_log(str(vehicle_id))} not found or not owned by user"
            )

        aggregates = await self.get_reminder_aggregates(db, [vehicle_id], date.today(), user_id)
        counts = aggregates.get(
            str(_as_uuid(vehicle_id)), {"overdue": 0, "completed_6mo": 0, "upcoming": 0}
        )

        score = compute_health_score(counts["overdue"], counts["completed_6mo"])
        category, color = categorize_health(score)

        return {
            "vehicle_id": vehicle_id,
            "score": score,
            "category": category,
            "category_color": color,
            "factors": build_health_factors(counts["overdue"], counts["completed_6mo"]),
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
            # Ownership check — prevent IDOR via vehicle_id
            vehicle_id = data.get("vehicle_id")
            if vehicle_id:
                vehicle = await self.get_vehicle(db, str(vehicle_id), user_id)
                if not vehicle:
                    raise VehicleGarageServiceError(
                        f"Vehicle {sanitize_log(str(vehicle_id))} not found or not owned by user"
                    )
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
            # Ownership check — prevent IDOR via vehicle_id
            vehicle_id = data.get("vehicle_id")
            if vehicle_id:
                vehicle = await self.get_vehicle(db, str(vehicle_id), user_id)
                if not vehicle:
                    raise VehicleGarageServiceError(
                        f"Vehicle {sanitize_log(str(vehicle_id))} not found or not owned by user"
                    )
            # Coerce Uuid-typed fields to real uuid.UUID (SQLite bind portability).
            cost_data = dict(data)
            if cost_data.get("vehicle_id") is not None:
                cost_data["vehicle_id"] = _as_uuid(cost_data["vehicle_id"])
            if cost_data.get("diagnosis_session_id") is not None:
                cost_data["diagnosis_session_id"] = _as_uuid(cost_data["diagnosis_session_id"])
            cost = MaintenanceCost(id=uuid4(), user_id=_as_uuid(user_id), **cost_data)
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
        conditions = [MaintenanceCost.user_id == _as_uuid(user_id)]
        if vehicle_id:
            conditions.append(MaintenanceCost.vehicle_id == _as_uuid(vehicle_id))

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
