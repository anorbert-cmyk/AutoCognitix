"""
Garage API endpoints — vehicle management, maintenance reminders, and cost tracking.
All endpoints require authentication.
"""

from datetime import date as date_type
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.endpoints.auth import get_current_user_from_token
from app.api.v1.schemas.garage import (
    MaintenanceCostCreate,
    MaintenanceCostListResponse,
    MaintenanceCostResponse,
    MaintenanceReminderCreate,
    MaintenanceReminderListResponse,
    MaintenanceReminderResponse,
    REMINDER_TYPE_LABELS,
    UserVehicleCreate,
    UserVehicleListResponse,
    UserVehicleResponse,
    UserVehicleUpdate,
    VehicleHealthScore,
)
from app.core.log_sanitizer import sanitize_log
from app.core.logging import get_logger
from app.db.postgres.models import MaintenanceReminder, User, UserVehicle
from app.db.postgres.session import get_db
from app.services.vehicle_garage_service import (
    VehicleGarageServiceError,
    get_vehicle_garage_service,
)

router = APIRouter()
logger = get_logger(__name__)


# =============================================================================
# Helpers
# =============================================================================


async def _get_vehicle_or_404(
    vehicle_id: str,
    user_id: str,
    db: AsyncSession,
) -> UserVehicle:
    service = get_vehicle_garage_service()
    vehicle = await service.get_vehicle(db, vehicle_id, user_id)
    if not vehicle:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {"message_hu": "A jármű nem található vagy nincs jogosultságod hozzá."}
            },
        )
    return vehicle


def _enrich_reminder(reminder: MaintenanceReminder) -> dict:
    today = date_type.today()
    days_until = None
    urgency = "ok"
    if reminder.due_date:
        days_until = (reminder.due_date - today).days
        if days_until < 0:
            urgency = "overdue"
        elif days_until <= 7:
            urgency = "urgent"
        elif days_until <= 30:
            urgency = "upcoming"
    return {
        "days_until_due": days_until,
        "urgency": urgency,
        "reminder_type_label": REMINDER_TYPE_LABELS.get(
            reminder.reminder_type, reminder.reminder_type
        ),
    }


def _build_reminder_response(reminder: MaintenanceReminder) -> MaintenanceReminderResponse:
    enriched = _enrich_reminder(reminder)
    data = {
        "id": str(reminder.id),
        "vehicle_id": str(reminder.vehicle_id),
        "user_id": str(reminder.user_id),
        "reminder_type": reminder.reminder_type,
        "reminder_type_label": enriched["reminder_type_label"],
        "title": reminder.title,
        "due_date": reminder.due_date,
        "due_mileage_km": reminder.due_mileage_km,
        "notes": reminder.notes,
        "is_completed": reminder.is_completed,
        "completed_at": reminder.completed_at,
        "email_sent_at": reminder.email_sent_at,
        "created_at": reminder.created_at,
        "updated_at": reminder.updated_at,
        "days_until_due": enriched["days_until_due"],
        "urgency": enriched["urgency"],
    }
    return MaintenanceReminderResponse(**data)


# =============================================================================
# Vehicle Endpoints
# =============================================================================


@router.get(
    "/vehicles",
    response_model=UserVehicleListResponse,
    status_code=status.HTTP_200_OK,
    summary="Járművek listázása",
    description="Az aktuális felhasználó összes járművének listája.",
)
async def list_vehicles(
    skip: int = Query(default=0, ge=0, description="Kihagyandó elemek száma"),
    limit: int = Query(default=20, ge=1, le=100, description="Visszaadandó elemek max. száma"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user_from_token),
) -> UserVehicleListResponse:
    """List all vehicles belonging to the current user."""
    try:
        service = get_vehicle_garage_service()
        vehicles, total = await service.get_vehicles(
            db, str(current_user.id), skip=skip, limit=limit
        )

        logger.info(
            "Járművek listázva",
            extra={
                "user_id": sanitize_log(str(current_user.id)),
                "count": total,
            },
        )

        vehicle_responses = [UserVehicleResponse.model_validate(v) for v in vehicles]
        return UserVehicleListResponse(vehicles=vehicle_responses, total=total)

    except VehicleGarageServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": {"message_hu": exc.message}},
        ) from exc
    except Exception:
        logger.error("Váratlan hiba a járművek listázásakor", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message_hu": "Váratlan szerverhiba történt. Kérjük próbálja újra később."
                }
            },
        )


@router.post(
    "/vehicles",
    response_model=UserVehicleResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Jármű hozzáadása",
    description="Új jármű felvétele a garázsba.",
)
async def create_vehicle(
    data: UserVehicleCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user_from_token),
) -> UserVehicleResponse:
    """Create a new vehicle for the current user."""
    MAX_VEHICLES_PER_USER = 20
    try:
        service = get_vehicle_garage_service()
        _, existing_total = await service.get_vehicles(db, str(current_user.id), skip=0, limit=1)
        if existing_total >= MAX_VEHICLES_PER_USER:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": {"message_hu": f"Maximum {MAX_VEHICLES_PER_USER} jármű tárolható."}
                },
            )
        vehicle = await service.create_vehicle(
            db, str(current_user.id), data.model_dump(exclude_none=True)
        )

        logger.info(
            "Jármű létrehozva",
            extra={
                "user_id": sanitize_log(str(current_user.id)),
                "vehicle_id": sanitize_log(str(vehicle.id)),
                "make": sanitize_log(data.make),
                "model": sanitize_log(data.model),
            },
        )

        return UserVehicleResponse.model_validate(vehicle)  # type: ignore[no-any-return]

    except HTTPException:
        raise
    except VehicleGarageServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": {"message_hu": exc.message}},
        ) from exc
    except Exception:
        logger.error("Váratlan hiba a jármű létrehozásakor", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message_hu": "Váratlan szerverhiba történt. Kérjük próbálja újra később."
                }
            },
        )


@router.get(
    "/vehicles/{vehicle_id}",
    response_model=UserVehicleResponse,
    status_code=status.HTTP_200_OK,
    summary="Jármű lekérdezése",
    description="Egy konkrét jármű adatainak lekérdezése.",
)
async def get_vehicle(
    vehicle_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user_from_token),
) -> UserVehicleResponse:
    """Get a single vehicle by ID (ownership enforced)."""
    vehicle = await _get_vehicle_or_404(vehicle_id, str(current_user.id), db)

    logger.info(
        "Jármű lekérdezve",
        extra={
            "user_id": sanitize_log(str(current_user.id)),
            "vehicle_id": sanitize_log(vehicle_id),
        },
    )

    return UserVehicleResponse.model_validate(vehicle)  # type: ignore[no-any-return]


@router.put(
    "/vehicles/{vehicle_id}",
    response_model=UserVehicleResponse,
    status_code=status.HTTP_200_OK,
    summary="Jármű frissítése",
    description="Jármű adatainak módosítása.",
)
async def update_vehicle(
    vehicle_id: str,
    data: UserVehicleUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user_from_token),
) -> UserVehicleResponse:
    """Update a vehicle (ownership enforced)."""
    await _get_vehicle_or_404(vehicle_id, str(current_user.id), db)

    try:
        service = get_vehicle_garage_service()
        updated = await service.update_vehicle(
            db, vehicle_id, str(current_user.id), data.model_dump(exclude_none=True)
        )

        logger.info(
            "Jármű frissítve",
            extra={
                "user_id": sanitize_log(str(current_user.id)),
                "vehicle_id": sanitize_log(vehicle_id),
            },
        )

        return UserVehicleResponse.model_validate(updated)  # type: ignore[no-any-return]

    except VehicleGarageServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": {"message_hu": exc.message}},
        ) from exc
    except Exception:
        logger.error("Váratlan hiba a jármű frissítésekor", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message_hu": "Váratlan szerverhiba történt. Kérjük próbálja újra később."
                }
            },
        )


@router.delete(
    "/vehicles/{vehicle_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Jármű törlése",
    description="Jármű soft-delete törlése a garázsból.",
)
async def delete_vehicle(
    vehicle_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user_from_token),
) -> None:
    """Soft-delete a vehicle (ownership enforced)."""
    await _get_vehicle_or_404(vehicle_id, str(current_user.id), db)

    try:
        service = get_vehicle_garage_service()
        await service.delete_vehicle(db, vehicle_id, str(current_user.id))

        logger.info(
            "Jármű törölve",
            extra={
                "user_id": sanitize_log(str(current_user.id)),
                "vehicle_id": sanitize_log(vehicle_id),
            },
        )

    except VehicleGarageServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": {"message_hu": exc.message}},
        ) from exc
    except Exception:
        logger.error("Váratlan hiba a jármű törlésekor", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message_hu": "Váratlan szerverhiba történt. Kérjük próbálja újra később."
                }
            },
        )


@router.get(
    "/vehicles/{vehicle_id}/health",
    response_model=VehicleHealthScore,
    status_code=status.HTTP_200_OK,
    summary="Jármű egészségi pontszám",
    description="A jármű karbantartási állapotának összesített értékelése.",
)
async def get_vehicle_health(
    vehicle_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user_from_token),
) -> VehicleHealthScore:
    """Get the health score for a vehicle (ownership enforced)."""
    await _get_vehicle_or_404(vehicle_id, str(current_user.id), db)

    try:
        service = get_vehicle_garage_service()
        result = await service.get_health_score(db, vehicle_id, str(current_user.id))

        logger.info(
            "Jármű egészségi pontszám lekérdezve",
            extra={
                "user_id": sanitize_log(str(current_user.id)),
                "vehicle_id": sanitize_log(vehicle_id),
                "score": sanitize_log(
                    str(result.get("score")) if result.get("score") is not None else ""
                ),
            },
        )

        return VehicleHealthScore(**result)

    except VehicleGarageServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": {"message_hu": exc.message}},
        ) from exc
    except Exception:
        logger.error("Váratlan hiba az egészségi pontszám lekérdezésekor", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message_hu": "Váratlan szerverhiba történt. Kérjük próbálja újra később."
                }
            },
        )


# =============================================================================
# Reminder Endpoints
# =============================================================================


@router.get(
    "/reminders/upcoming",
    response_model=MaintenanceReminderListResponse,
    status_code=status.HTTP_200_OK,
    summary="Közelgő emlékeztetők",
    description="A következő N napon belül esedékes emlékeztetők listája.",
)
async def get_upcoming_reminders(
    days_ahead: int = Query(default=30, ge=1, le=365, description="Hány napra előre nézzen"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user_from_token),
) -> MaintenanceReminderListResponse:
    """Get upcoming reminders within days_ahead days."""
    try:
        service = get_vehicle_garage_service()
        reminders = await service.get_upcoming_reminders(
            db, str(current_user.id), days_ahead=days_ahead
        )

        logger.info(
            "Közelgő emlékeztetők lekérdezve",
            extra={
                "user_id": sanitize_log(str(current_user.id)),
                "days_ahead": sanitize_log(str(days_ahead)),
                "count": len(reminders),
            },
        )

        enriched: List[MaintenanceReminderResponse] = [
            _build_reminder_response(r) for r in reminders
        ]
        overdue_count = sum(1 for r in enriched if r.urgency == "overdue")
        urgent_count = sum(1 for r in enriched if r.urgency == "urgent")

        return MaintenanceReminderListResponse(
            reminders=enriched,
            total=len(enriched),
            overdue_count=overdue_count,
            urgent_count=urgent_count,
        )

    except VehicleGarageServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": {"message_hu": exc.message}},
        ) from exc
    except Exception:
        logger.error("Váratlan hiba a közelgő emlékeztetők lekérdezésekor", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message_hu": "Váratlan szerverhiba történt. Kérjük próbálja újra később."
                }
            },
        )


@router.get(
    "/reminders",
    response_model=MaintenanceReminderListResponse,
    status_code=status.HTTP_200_OK,
    summary="Emlékeztetők listázása",
    description="Az aktuális felhasználó emlékeztetőinek listája.",
)
async def list_reminders(
    vehicle_id: Optional[str] = Query(default=None, description="Szűrés jármű azonosítóra"),
    include_completed: bool = Query(default=False, description="Teljesített emlékeztetők is"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user_from_token),
) -> MaintenanceReminderListResponse:
    """List all reminders for the current user."""
    try:
        service = get_vehicle_garage_service()
        reminders, total = await service.get_reminders(
            db,
            str(current_user.id),
            vehicle_id=vehicle_id,
            include_completed=include_completed,
        )

        logger.info(
            "Emlékeztetők listázva",
            extra={
                "user_id": sanitize_log(str(current_user.id)),
                "vehicle_id": sanitize_log(vehicle_id) if vehicle_id else None,
                "count": total,
            },
        )

        enriched = [_build_reminder_response(r) for r in reminders]
        overdue_count = sum(1 for r in enriched if r.urgency == "overdue")
        urgent_count = sum(1 for r in enriched if r.urgency == "urgent")

        return MaintenanceReminderListResponse(
            reminders=enriched,
            total=total,
            overdue_count=overdue_count,
            urgent_count=urgent_count,
        )

    except VehicleGarageServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": {"message_hu": exc.message}},
        ) from exc
    except Exception:
        logger.error("Váratlan hiba az emlékeztetők listázásakor", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message_hu": "Váratlan szerverhiba történt. Kérjük próbálja újra később."
                }
            },
        )


@router.post(
    "/reminders",
    response_model=MaintenanceReminderResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Emlékeztető létrehozása",
    description="Új karbantartási emlékeztető felvétele.",
)
async def create_reminder(
    data: MaintenanceReminderCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user_from_token),
) -> MaintenanceReminderResponse:
    """Create a new maintenance reminder."""
    try:
        service = get_vehicle_garage_service()
        reminder = await service.create_reminder(
            db, str(current_user.id), data.model_dump(exclude_none=True)
        )

        logger.info(
            "Emlékeztető létrehozva",
            extra={
                "user_id": sanitize_log(str(current_user.id)),
                "reminder_id": sanitize_log(str(reminder.id)),
                "reminder_type": sanitize_log(reminder.reminder_type),
            },
        )

        return _build_reminder_response(reminder)

    except VehicleGarageServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": {"message_hu": exc.message}},
        ) from exc
    except Exception:
        logger.error("Váratlan hiba az emlékeztető létrehozásakor", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message_hu": "Váratlan szerverhiba történt. Kérjük próbálja újra később."
                }
            },
        )


@router.post(
    "/reminders/{reminder_id}/complete",
    response_model=MaintenanceReminderResponse,
    status_code=status.HTTP_200_OK,
    summary="Emlékeztető teljesítve",
    description="Emlékeztető megjelölése teljesítettként.",
)
async def complete_reminder(
    reminder_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user_from_token),
) -> MaintenanceReminderResponse:
    """Mark a reminder as completed."""
    try:
        service = get_vehicle_garage_service()
        reminder = await service.complete_reminder(db, reminder_id, str(current_user.id))

        if not reminder:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": {
                        "message_hu": "Az emlékeztető nem található vagy nincs jogosultságod hozzá."
                    }
                },
            )

        logger.info(
            "Emlékeztető teljesítve",
            extra={
                "user_id": sanitize_log(str(current_user.id)),
                "reminder_id": sanitize_log(reminder_id),
            },
        )

        return _build_reminder_response(reminder)

    except HTTPException:
        raise
    except VehicleGarageServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": {"message_hu": exc.message}},
        ) from exc
    except Exception:
        logger.error("Váratlan hiba az emlékeztető teljesítésekor", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message_hu": "Váratlan szerverhiba történt. Kérjük próbálja újra később."
                }
            },
        )


@router.delete(
    "/reminders/{reminder_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Emlékeztető törlése",
    description="Emlékeztető törlése.",
)
async def delete_reminder(
    reminder_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user_from_token),
) -> None:
    """Delete a reminder."""
    try:
        service = get_vehicle_garage_service()
        await service.delete_reminder(db, reminder_id, str(current_user.id))

        logger.info(
            "Emlékeztető törölve",
            extra={
                "user_id": sanitize_log(str(current_user.id)),
                "reminder_id": sanitize_log(reminder_id),
            },
        )

    except VehicleGarageServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": {"message_hu": exc.message}},
        ) from exc
    except Exception:
        logger.error("Váratlan hiba az emlékeztető törlésekor", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message_hu": "Váratlan szerverhiba történt. Kérjük próbálja újra később."
                }
            },
        )


# =============================================================================
# Cost Endpoints
# =============================================================================


@router.get(
    "/costs",
    response_model=MaintenanceCostListResponse,
    status_code=status.HTTP_200_OK,
    summary="Karbantartási költségek listázása",
    description="Az aktuális felhasználó karbantartási költségeinek listája.",
)
async def list_costs(
    vehicle_id: Optional[str] = Query(default=None, description="Szűrés jármű azonosítóra"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user_from_token),
) -> MaintenanceCostListResponse:
    """List all maintenance costs for the current user."""
    try:
        service = get_vehicle_garage_service()
        costs, total, total_cost_huf = await service.get_costs(
            db, str(current_user.id), vehicle_id=vehicle_id
        )

        logger.info(
            "Karbantartási költségek listázva",
            extra={
                "user_id": sanitize_log(str(current_user.id)),
                "vehicle_id": sanitize_log(vehicle_id) if vehicle_id else None,
                "count": total,
                "total_cost_huf": total_cost_huf,
            },
        )

        cost_responses = [MaintenanceCostResponse.model_validate(c) for c in costs]
        return MaintenanceCostListResponse(
            costs=cost_responses,
            total=total,
            total_cost_huf=total_cost_huf,
        )

    except VehicleGarageServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": {"message_hu": exc.message}},
        ) from exc
    except Exception:
        logger.error("Váratlan hiba a karbantartási költségek listázásakor", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message_hu": "Váratlan szerverhiba történt. Kérjük próbálja újra később."
                }
            },
        )


@router.post(
    "/costs",
    response_model=MaintenanceCostResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Karbantartási költség rögzítése",
    description="Új karbantartási költség bejegyzés felvétele.",
)
async def create_cost(
    data: MaintenanceCostCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user_from_token),
) -> MaintenanceCostResponse:
    """Create a new maintenance cost entry."""
    try:
        service = get_vehicle_garage_service()
        cost = await service.create_cost(
            db, str(current_user.id), data.model_dump(exclude_none=True)
        )

        logger.info(
            "Karbantartási költség rögzítve",
            extra={
                "user_id": sanitize_log(str(current_user.id)),
                "cost_id": sanitize_log(str(cost.id)),
                "cost_huf": sanitize_log(str(data.cost_huf)),
            },
        )

        return MaintenanceCostResponse.model_validate(cost)  # type: ignore[no-any-return]

    except VehicleGarageServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": {"message_hu": exc.message}},
        ) from exc
    except Exception:
        logger.error("Váratlan hiba a karbantartási költség rögzítésekor", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message_hu": "Váratlan szerverhiba történt. Kérjük próbálja újra később."
                }
            },
        )


@router.get(
    "/vehicles/{vehicle_id}/recalls",
    response_model=List[dict],
    status_code=status.HTTP_200_OK,
    summary="Jármű visszahívások",
    description="NHTSA visszahívások lekérdezése a jármű gyártója/modellje/évjárata alapján.",
)
async def get_vehicle_recalls(
    vehicle_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user_from_token),
) -> List[dict]:
    """Fetch NHTSA recalls for a vehicle owned by the current user."""
    vehicle = await _get_vehicle_or_404(vehicle_id, str(current_user.id), db)

    try:
        from app.services.nhtsa_service import get_nhtsa_service

        nhtsa = await get_nhtsa_service()
        recalls = await nhtsa.get_recalls(
            make=vehicle.make,
            model=vehicle.model,
            year=vehicle.year,
        )

        logger.info(
            "Visszahívások lekérdezve",
            extra={
                "user_id": sanitize_log(str(current_user.id)),
                "vehicle_id": sanitize_log(vehicle_id),
                "recall_count": len(recalls),
            },
        )

        return [r.model_dump() for r in recalls]

    except Exception:
        logger.warning("Visszahívások lekérdezése sikertelen", exc_info=True)
        return []
