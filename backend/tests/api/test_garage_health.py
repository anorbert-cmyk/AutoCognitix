"""
Tests for garage health-score aggregation (Sprint 2, item 0.2).

Covers:
- Pure scoring helpers: compute_health_score / build_health_factors / categorize_health
- VehicleGarageService.get_reminder_aggregates grouped per-vehicle counts
- get_health_score parity with the pre-refactor behaviour
- GET /api/v1/garage/vehicles returning REAL health_score / upcoming_reminders_count
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from app.db.postgres.models import MaintenanceCost, MaintenanceReminder, UserVehicle
from app.services.vehicle_garage_service import (
    VehicleGarageService,
    build_health_factors,
    categorize_health,
    compute_health_score,
    get_vehicle_garage_service,
)

if TYPE_CHECKING:
    from httpx import AsyncClient
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.db.postgres.models import User


# =============================================================================
# Helpers
# =============================================================================


def _utc(days_offset: int) -> datetime:
    """A timezone-aware UTC datetime `days_offset` days from now."""
    return datetime.now(timezone.utc) + timedelta(days=days_offset)


async def _add_vehicle(db: AsyncSession, user: User, **overrides) -> UserVehicle:
    """Insert (and flush) a UserVehicle owned by `user`."""
    now = datetime.now(timezone.utc)
    vehicle = UserVehicle(
        id=uuid4(),
        user_id=user.id,
        make=overrides.pop("make", "Volkswagen"),
        model=overrides.pop("model", "Golf"),
        year=overrides.pop("year", 2018),
        is_active=True,
        created_at=now,
        updated_at=now,
        **overrides,
    )
    db.add(vehicle)
    await db.flush()
    return vehicle


async def _add_reminder(
    db: AsyncSession, vehicle: UserVehicle, user: User, **overrides
) -> MaintenanceReminder:
    """Insert (and flush) a MaintenanceReminder for `vehicle`."""
    reminder = MaintenanceReminder(
        id=uuid4(),
        vehicle_id=vehicle.id,
        user_id=user.id,
        reminder_type=overrides.pop("reminder_type", "oil_change"),
        title=overrides.pop("title", "Olajcsere"),
        **overrides,
    )
    db.add(reminder)
    await db.flush()
    return reminder


async def _add_cost(
    db: AsyncSession, vehicle: UserVehicle, user: User, **overrides
) -> MaintenanceCost:
    """Insert (and flush) a MaintenanceCost for `vehicle`."""
    cost = MaintenanceCost(
        id=uuid4(),
        vehicle_id=vehicle.id,
        user_id=user.id,
        service_type=overrides.pop("service_type", "Olajcsere"),
        cost_huf=overrides.pop("cost_huf", 25000),
        service_date=overrides.pop("service_date", date.today()),
        **overrides,
    )
    db.add(cost)
    await db.flush()
    return cost


@pytest.fixture
def service() -> VehicleGarageService:
    return get_vehicle_garage_service()


# =============================================================================
# Pure scoring helpers
# =============================================================================


class TestComputeHealthScore:
    @pytest.mark.parametrize(
        ("overdue", "completed", "expected"),
        [
            (0, 0, 80),
            (1, 0, 70),
            (5, 0, 40),
            (0, 3, 89),
            (0, 10, 95),
            (2, 1, 63),
        ],
    )
    def test_expected_scores(self, overdue: int, completed: int, expected: int):
        assert compute_health_score(overdue, completed) == expected

    def test_penalty_and_bonus_are_capped(self):
        # Overdue penalty caps at -40 → never below 40 from overdue alone.
        assert compute_health_score(4, 0) == 40
        assert compute_health_score(100, 0) == 40
        # Completed bonus caps at +15.
        assert compute_health_score(0, 1000) == 95

    def test_result_stays_within_reachable_domain(self):
        # Base 80; overdue caps at -40 (floor 40) and completed caps at +15 (ceil 95),
        # so the reachable domain is [40, 95] — a stronger invariant than the 0..100 clamp.
        for overdue in range(0, 40):
            for completed in range(0, 40):
                score = compute_health_score(overdue, completed)
                assert 40 <= score <= 95


class TestBuildHealthFactors:
    def test_no_factors_when_clean(self):
        assert build_health_factors(0, 0) == []

    def test_negative_factor(self):
        factors = build_health_factors(2, 0)
        assert len(factors) == 1
        assert factors[0]["type"] == "negative"
        assert factors[0]["impact"] == -20
        assert "2" in factors[0]["label"]

    def test_positive_factor(self):
        factors = build_health_factors(0, 3)
        assert len(factors) == 1
        assert factors[0]["type"] == "positive"
        assert factors[0]["impact"] == 9

    def test_penalty_and_bonus_capped_in_factors(self):
        assert build_health_factors(10, 10) == [
            {"type": "negative", "label": "10 lejárt emlékeztető", "impact": -40},
            {
                "type": "positive",
                "label": "10 elvégzett karbantartás (6 hónap)",
                "impact": 15,
            },
        ]

    def test_factor_order_is_penalty_then_bonus(self):
        factors = build_health_factors(1, 1)
        assert [f["type"] for f in factors] == ["negative", "positive"]


class TestCategorizeHealth:
    @pytest.mark.parametrize(
        ("score", "category", "color"),
        [
            (100, "kiváló", "green"),
            (80, "kiváló", "green"),
            (79, "jó", "yellow"),
            (60, "jó", "yellow"),
            (59, "figyelmet_igényel", "orange"),
            (40, "figyelmet_igényel", "orange"),
            (39, "kritikus", "red"),
            (0, "kritikus", "red"),
        ],
    )
    def test_thresholds(self, score: int, category: str, color: str):
        assert categorize_health(score) == (category, color)


# =============================================================================
# get_reminder_aggregates
# =============================================================================


class TestGetReminderAggregates:
    @pytest.mark.asyncio
    async def test_empty_ids_short_circuits(
        self, db_session: AsyncSession, service: VehicleGarageService
    ):
        assert (
            await service.get_reminder_aggregates(db_session, [], date.today(), str(uuid4())) == {}
        )

    @pytest.mark.asyncio
    async def test_vehicle_absent_when_no_reminders(
        self, db_session: AsyncSession, test_user: User, service: VehicleGarageService
    ):
        vehicle = await _add_vehicle(db_session, test_user)
        result = await service.get_reminder_aggregates(
            db_session, [str(vehicle.id)], date.today(), str(test_user.id)
        )
        assert result == {}
        assert str(vehicle.id) not in result

    @pytest.mark.asyncio
    async def test_mixed_rows(
        self, db_session: AsyncSession, test_user: User, service: VehicleGarageService
    ):
        today = date.today()
        vehicle = await _add_vehicle(db_session, test_user)
        # Overdue (also counts as upcoming — due date is on/before the window).
        await _add_reminder(
            db_session, vehicle, test_user, due_date=today - timedelta(days=5), is_completed=False
        )
        # Upcoming within the 30-day window.
        await _add_reminder(
            db_session, vehicle, test_user, due_date=today + timedelta(days=10), is_completed=False
        )
        # Beyond the upcoming window (+60d) → excluded from every bucket.
        await _add_reminder(
            db_session, vehicle, test_user, due_date=today + timedelta(days=60), is_completed=False
        )
        # Completed within 6 months → counts towards completed_6mo.
        await _add_reminder(
            db_session, vehicle, test_user, is_completed=True, completed_at=_utc(-30)
        )
        # Completed 240 days ago → outside the 6-month boundary → excluded.
        await _add_reminder(
            db_session, vehicle, test_user, is_completed=True, completed_at=_utc(-240)
        )

        result = await service.get_reminder_aggregates(
            db_session, [str(vehicle.id)], today, str(test_user.id)
        )
        assert result[str(vehicle.id)] == {"overdue": 1, "completed_6mo": 1, "upcoming": 2}

    @pytest.mark.asyncio
    async def test_upcoming_window_boundaries(
        self, db_session: AsyncSession, test_user: User, service: VehicleGarageService
    ):
        today = date.today()
        vehicle = await _add_vehicle(db_session, test_user)
        # due == today → upcoming, never overdue (overdue is a strict due_date < today).
        await _add_reminder(db_session, vehicle, test_user, due_date=today, is_completed=False)
        # due == today + 30 → inclusive upper bound of the window → upcoming.
        await _add_reminder(
            db_session, vehicle, test_user, due_date=today + timedelta(days=30), is_completed=False
        )
        # due == today + 31 → one day past the window → excluded from every bucket.
        await _add_reminder(
            db_session, vehicle, test_user, due_date=today + timedelta(days=31), is_completed=False
        )

        result = await service.get_reminder_aggregates(
            db_session, [str(vehicle.id)], today, str(test_user.id)
        )
        assert result[str(vehicle.id)] == {"overdue": 0, "completed_6mo": 0, "upcoming": 2}

    @pytest.mark.asyncio
    async def test_two_vehicles_do_not_bleed(
        self, db_session: AsyncSession, test_user: User, service: VehicleGarageService
    ):
        today = date.today()
        v1 = await _add_vehicle(db_session, test_user, model="Golf")
        v2 = await _add_vehicle(db_session, test_user, model="Passat")
        await _add_reminder(
            db_session, v1, test_user, due_date=today - timedelta(days=3), is_completed=False
        )
        await _add_reminder(db_session, v2, test_user, is_completed=True, completed_at=_utc(-10))

        result = await service.get_reminder_aggregates(
            db_session, [str(v1.id), str(v2.id)], today, str(test_user.id)
        )
        assert result[str(v1.id)] == {"overdue": 1, "completed_6mo": 0, "upcoming": 1}
        assert result[str(v2.id)] == {"overdue": 0, "completed_6mo": 1, "upcoming": 0}


# =============================================================================
# get_health_score (parity with pre-refactor behaviour)
# =============================================================================


class TestGetHealthScoreParity:
    @pytest.mark.asyncio
    async def test_one_overdue_scores_70(
        self, db_session: AsyncSession, test_user: User, service: VehicleGarageService
    ):
        vehicle = await _add_vehicle(db_session, test_user)
        await _add_reminder(
            db_session,
            vehicle,
            test_user,
            due_date=date.today() - timedelta(days=5),
            is_completed=False,
        )

        result = await service.get_health_score(db_session, str(vehicle.id), str(test_user.id))

        assert set(result.keys()) == {
            "vehicle_id",
            "score",
            "category",
            "category_color",
            "factors",
        }
        assert result["score"] == 70
        assert result["category"] == "jó"
        assert result["category_color"] == "yellow"
        negatives = [f for f in result["factors"] if f["type"] == "negative"]
        assert len(negatives) == 1
        assert negatives[0]["impact"] == -10

    @pytest.mark.asyncio
    async def test_unknown_vehicle_raises(
        self, db_session: AsyncSession, test_user: User, service: VehicleGarageService
    ):
        from app.services.vehicle_garage_service import VehicleGarageServiceError

        with pytest.raises(VehicleGarageServiceError):
            await service.get_health_score(db_session, str(uuid4()), str(test_user.id))


# =============================================================================
# GET /api/v1/garage/vehicles endpoint
# =============================================================================


class TestListVehiclesEndpoint:
    @pytest.mark.asyncio
    async def test_real_health_and_upcoming_counts(
        self, authenticated_client: AsyncClient, db_session: AsyncSession, test_user: User
    ):
        today = date.today()
        vehicle = await _add_vehicle(db_session, test_user)
        # Overdue (counts once as overdue and once as upcoming).
        await _add_reminder(
            db_session, vehicle, test_user, due_date=today - timedelta(days=5), is_completed=False
        )
        # A second, upcoming reminder.
        await _add_reminder(
            db_session, vehicle, test_user, due_date=today + timedelta(days=10), is_completed=False
        )
        await db_session.flush()

        response = await authenticated_client.get("/api/v1/garage/vehicles")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        v = data["vehicles"][0]
        assert v["health_score"] == 70
        assert v["upcoming_reminders_count"] == 2
        assert v["health_score"] is not None
        assert isinstance(v["health_score"], int)
        assert isinstance(v["upcoming_reminders_count"], int)

    @pytest.mark.asyncio
    async def test_vehicle_without_reminders_scores_80_zero(
        self, authenticated_client: AsyncClient, db_session: AsyncSession, test_user: User
    ):
        await _add_vehicle(db_session, test_user)
        await db_session.flush()

        response = await authenticated_client.get("/api/v1/garage/vehicles")

        assert response.status_code == 200
        v = response.json()["vehicles"][0]
        assert v["health_score"] == 80
        assert v["upcoming_reminders_count"] == 0

    @pytest.mark.asyncio
    async def test_zero_vehicles_returns_empty_200(self, authenticated_client: AsyncClient):
        response = await authenticated_client.get("/api/v1/garage/vehicles")

        assert response.status_code == 200
        data = response.json()
        assert data["vehicles"] == []
        assert data["total"] == 0


# =============================================================================
# Previously UUID→str-broken endpoints (create/get/update vehicle, costs)
# =============================================================================


class TestUUIDStrResponseEndpoints:
    """Happy-path 2xx coverage for endpoints that return ``Response.model_validate(orm)``.

    The ORM yields ``uuid.UUID`` for the ``Uuid`` id columns; before the
    ``UUIDStrModel`` before-validator these raised a pydantic ValidationError on the
    happy path → 500. Each response id must now serialize as a plain ``str``.
    """

    @pytest.mark.asyncio
    async def test_create_vehicle_returns_201_with_str_id(self, authenticated_client: AsyncClient):
        response = await authenticated_client.post(
            "/api/v1/garage/vehicles",
            json={"make": "Volkswagen", "model": "Golf", "year": 2018},
        )

        assert response.status_code == 201
        data = response.json()
        assert isinstance(data["id"], str)
        assert isinstance(data["user_id"], str)
        assert data["make"] == "Volkswagen"

    @pytest.mark.asyncio
    async def test_get_vehicle_returns_200(
        self, authenticated_client: AsyncClient, db_session: AsyncSession, test_user: User
    ):
        vehicle = await _add_vehicle(db_session, test_user)
        await db_session.flush()

        response = await authenticated_client.get(f"/api/v1/garage/vehicles/{vehicle.id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(vehicle.id)
        assert isinstance(data["id"], str)

    @pytest.mark.asyncio
    async def test_update_vehicle_returns_200(
        self, authenticated_client: AsyncClient, db_session: AsyncSession, test_user: User
    ):
        vehicle = await _add_vehicle(db_session, test_user)
        await db_session.flush()

        response = await authenticated_client.put(
            f"/api/v1/garage/vehicles/{vehicle.id}",
            json={"nickname": "Az én Golfom"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["nickname"] == "Az én Golfom"
        assert isinstance(data["id"], str)

    @pytest.mark.asyncio
    async def test_create_cost_returns_201_with_str_id(
        self, authenticated_client: AsyncClient, db_session: AsyncSession, test_user: User
    ):
        vehicle = await _add_vehicle(db_session, test_user)
        await db_session.flush()

        response = await authenticated_client.post(
            "/api/v1/garage/costs",
            json={
                "vehicle_id": str(vehicle.id),
                "service_type": "Olajcsere",
                "cost_huf": 25000,
                "service_date": "2026-01-15",
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert isinstance(data["id"], str)
        assert data["vehicle_id"] == str(vehicle.id)
        assert data["cost_huf"] == 25000

    @pytest.mark.asyncio
    async def test_list_costs_returns_200(
        self, authenticated_client: AsyncClient, db_session: AsyncSession, test_user: User
    ):
        vehicle = await _add_vehicle(db_session, test_user)
        await _add_cost(db_session, vehicle, test_user)
        await db_session.flush()

        response = await authenticated_client.get("/api/v1/garage/costs")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert isinstance(data["costs"][0]["id"], str)


class TestMalformedVehicleId:
    @pytest.mark.asyncio
    async def test_malformed_uuid_returns_404(self, authenticated_client: AsyncClient):
        # A non-UUID path param must resolve to 404, not a 500 from UUID() parsing.
        response = await authenticated_client.get("/api/v1/garage/vehicles/not-a-uuid")
        assert response.status_code == 404
