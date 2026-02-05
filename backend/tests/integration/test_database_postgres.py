"""
Integration tests for PostgreSQL database operations.

Tests repository CRUD operations with actual database connections.
"""

import pytest
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(backend_path))

from app.db.postgres.repositories import (
    DTCCodeRepository,
    UserRepository,
    DiagnosisSessionRepository,
    VehicleMakeRepository,
    VehicleModelRepository,
)


class TestDTCCodeRepository:
    """Test DTCCode repository operations."""

    @pytest.mark.asyncio
    async def test_create_dtc_code(self, db_session):
        """Test creating a new DTC code."""
        repo = DTCCodeRepository(db_session)

        dtc_data = {
            "code": "P0300",
            "description_en": "Random/Multiple Cylinder Misfire Detected",
            "description_hu": "Veletlenszeru/tobbszoros hengergyujtasi hiba",
            "category": "powertrain",
            "severity": "high",
            "is_generic": True,
        }

        dtc = await repo.create(dtc_data)

        assert dtc.id is not None
        assert dtc.code == "P0300"
        assert dtc.category == "powertrain"

    @pytest.mark.asyncio
    async def test_get_dtc_by_code(self, db_session, seeded_db):
        """Test getting DTC by code."""
        repo = DTCCodeRepository(db_session)

        dtc = await repo.get_by_code("P0101")

        assert dtc is not None
        assert dtc.code == "P0101"
        assert dtc.category == "powertrain"

    @pytest.mark.asyncio
    async def test_get_dtc_by_code_case_insensitive(self, db_session, seeded_db):
        """Test that code lookup is case-insensitive."""
        repo = DTCCodeRepository(db_session)

        dtc = await repo.get_by_code("p0101")  # lowercase

        assert dtc is not None
        assert dtc.code == "P0101"

    @pytest.mark.asyncio
    async def test_get_dtc_by_code_not_found(self, db_session, seeded_db):
        """Test getting nonexistent DTC code returns None."""
        repo = DTCCodeRepository(db_session)

        dtc = await repo.get_by_code("P9999")

        assert dtc is None

    @pytest.mark.asyncio
    async def test_search_dtc_by_code(self, db_session, seeded_db):
        """Test searching DTC codes by code."""
        repo = DTCCodeRepository(db_session)

        results = await repo.search("P01")

        assert len(results) >= 1
        codes = [r.code for r in results]
        assert "P0101" in codes or "P0171" in codes

    @pytest.mark.asyncio
    async def test_search_dtc_by_description(self, db_session, seeded_db):
        """Test searching DTC codes by description."""
        repo = DTCCodeRepository(db_session)

        results = await repo.search("mass air")

        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_search_dtc_with_category_filter(self, db_session, seeded_db):
        """Test searching with category filter."""
        repo = DTCCodeRepository(db_session)

        results = await repo.search("0", category="powertrain")

        for result in results:
            assert result.category == "powertrain"

    @pytest.mark.asyncio
    async def test_search_dtc_with_limit(self, db_session, seeded_db):
        """Test search respects limit."""
        repo = DTCCodeRepository(db_session)

        results = await repo.search("P", limit=2)

        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_get_related_codes(self, db_session, seeded_db):
        """Test getting related DTC codes."""
        repo = DTCCodeRepository(db_session)

        # First, make sure we have a code with related codes
        dtc = await repo.get_by_code("P0101")
        if dtc and dtc.related_codes:
            related = await repo.get_related_codes("P0101")
            # Should return empty if related codes don't exist in DB
            assert isinstance(related, list)

    @pytest.mark.asyncio
    async def test_update_dtc_code(self, db_session, seeded_db):
        """Test updating a DTC code."""
        repo = DTCCodeRepository(db_session)

        # Get existing code
        dtc = await repo.get_by_code("P0101")
        assert dtc is not None

        # Update severity
        updated = await repo.update(dtc.id, {"severity": "critical"})

        assert updated is not None
        assert updated.severity == "critical"

    @pytest.mark.asyncio
    async def test_delete_dtc_code(self, db_session):
        """Test deleting a DTC code."""
        repo = DTCCodeRepository(db_session)

        # Create a code to delete
        dtc = await repo.create({
            "code": "P9998",
            "description_en": "Test code to delete",
            "category": "powertrain",
        })

        # Delete it
        result = await repo.delete(dtc.id)

        assert result is True

        # Verify it's gone
        deleted = await repo.get(dtc.id)
        assert deleted is None


class TestUserRepository:
    """Test User repository operations."""

    @pytest.mark.asyncio
    async def test_create_user(self, db_session):
        """Test creating a new user."""
        repo = UserRepository(db_session)

        user_data = {
            "email": "newuser@example.com",
            "hashed_password": "$2b$12$hashedpassword",
            "full_name": "New User",
        }

        user = await repo.create(user_data)

        assert user.id is not None
        assert user.email == "newuser@example.com"
        assert user.is_active is True
        assert user.role == "user"

    @pytest.mark.asyncio
    async def test_get_user_by_email(self, db_session, seeded_db):
        """Test getting user by email."""
        repo = UserRepository(db_session)

        user = await repo.get_by_email("test@example.com")

        assert user is not None
        assert user.email == "test@example.com"

    @pytest.mark.asyncio
    async def test_get_user_by_email_not_found(self, db_session, seeded_db):
        """Test getting nonexistent user returns None."""
        repo = UserRepository(db_session)

        user = await repo.get_by_email("nonexistent@example.com")

        assert user is None

    @pytest.mark.asyncio
    async def test_get_user_by_id(self, db_session, seeded_db):
        """Test getting user by ID."""
        repo = UserRepository(db_session)

        # First get by email to find ID
        user = await repo.get_by_email("test@example.com")
        assert user is not None

        # Then get by ID
        user_by_id = await repo.get(user.id)

        assert user_by_id is not None
        assert user_by_id.email == "test@example.com"

    @pytest.mark.asyncio
    async def test_update_user(self, db_session, seeded_db):
        """Test updating a user."""
        repo = UserRepository(db_session)

        user = await repo.get_by_email("test@example.com")
        assert user is not None

        updated = await repo.update(user.id, {"full_name": "Updated Name"})

        assert updated is not None
        assert updated.full_name == "Updated Name"

    @pytest.mark.asyncio
    async def test_deactivate_user(self, db_session, seeded_db):
        """Test deactivating a user."""
        repo = UserRepository(db_session)

        user = await repo.get_by_email("test@example.com")
        assert user is not None

        updated = await repo.update(user.id, {"is_active": False})

        assert updated is not None
        assert updated.is_active is False


class TestDiagnosisSessionRepository:
    """Test DiagnosisSession repository operations."""

    @pytest.mark.asyncio
    async def test_create_diagnosis_session(self, db_session, seeded_db):
        """Test creating a new diagnosis session."""
        repo = DiagnosisSessionRepository(db_session)

        session_data = {
            "vehicle_make": "Volkswagen",
            "vehicle_model": "Golf",
            "vehicle_year": 2018,
            "dtc_codes": ["P0101", "P0171"],
            "symptoms_text": "Motor nehezen indul",
            "diagnosis_result": {"probable_causes": []},
            "confidence_score": 0.75,
        }

        session = await repo.create(session_data)

        assert session.id is not None
        assert session.vehicle_make == "Volkswagen"
        assert session.dtc_codes == ["P0101", "P0171"]

    @pytest.mark.asyncio
    async def test_create_diagnosis_session_with_user(self, db_session, seeded_db):
        """Test creating diagnosis session linked to user."""
        user_repo = UserRepository(db_session)
        user = await user_repo.get_by_email("test@example.com")

        repo = DiagnosisSessionRepository(db_session)

        session_data = {
            "user_id": user.id,
            "vehicle_make": "Volkswagen",
            "vehicle_model": "Golf",
            "vehicle_year": 2018,
            "dtc_codes": ["P0101"],
            "symptoms_text": "Motor nehezen indul",
            "diagnosis_result": {"probable_causes": []},
            "confidence_score": 0.75,
        }

        session = await repo.create(session_data)

        assert session.user_id == user.id

    @pytest.mark.asyncio
    async def test_get_diagnosis_by_id(self, db_session, seeded_db):
        """Test getting diagnosis session by ID."""
        repo = DiagnosisSessionRepository(db_session)

        # Create a session first
        session = await repo.create({
            "vehicle_make": "Volkswagen",
            "vehicle_model": "Golf",
            "vehicle_year": 2018,
            "dtc_codes": ["P0101"],
            "symptoms_text": "Motor nehezen indul",
            "diagnosis_result": {},
            "confidence_score": 0.75,
        })

        # Get it by ID
        retrieved = await repo.get(session.id)

        assert retrieved is not None
        assert retrieved.vehicle_make == "Volkswagen"

    @pytest.mark.asyncio
    async def test_get_user_history(self, db_session, seeded_db):
        """Test getting diagnosis history for a user."""
        user_repo = UserRepository(db_session)
        user = await user_repo.get_by_email("test@example.com")

        repo = DiagnosisSessionRepository(db_session)

        # Create some sessions for the user
        for i in range(3):
            await repo.create({
                "user_id": user.id,
                "vehicle_make": "Volkswagen",
                "vehicle_model": "Golf",
                "vehicle_year": 2018 + i,
                "dtc_codes": [f"P010{i}"],
                "symptoms_text": f"Problem {i}",
                "diagnosis_result": {},
                "confidence_score": 0.75,
            })

        # Get history
        history = await repo.get_user_history(user.id)

        assert len(history) == 3

    @pytest.mark.asyncio
    async def test_get_user_history_with_pagination(self, db_session, seeded_db):
        """Test user history pagination."""
        user_repo = UserRepository(db_session)
        user = await user_repo.get_by_email("test@example.com")

        repo = DiagnosisSessionRepository(db_session)

        # Create sessions
        for i in range(5):
            await repo.create({
                "user_id": user.id,
                "vehicle_make": "Volkswagen",
                "vehicle_model": "Golf",
                "vehicle_year": 2018,
                "dtc_codes": ["P0101"],
                "symptoms_text": f"Problem {i}",
                "diagnosis_result": {},
                "confidence_score": 0.75,
            })

        # Get with pagination
        page1 = await repo.get_user_history(user.id, skip=0, limit=2)
        page2 = await repo.get_user_history(user.id, skip=2, limit=2)

        assert len(page1) == 2
        assert len(page2) == 2

    @pytest.mark.asyncio
    async def test_get_user_history_ordered_by_date(self, db_session, seeded_db):
        """Test that user history is ordered by date descending."""
        user_repo = UserRepository(db_session)
        user = await user_repo.get_by_email("test@example.com")

        repo = DiagnosisSessionRepository(db_session)

        # Create sessions
        for i in range(3):
            await repo.create({
                "user_id": user.id,
                "vehicle_make": "Volkswagen",
                "vehicle_model": "Golf",
                "vehicle_year": 2018,
                "dtc_codes": ["P0101"],
                "symptoms_text": f"Problem {i}",
                "diagnosis_result": {},
                "confidence_score": 0.75,
            })

        history = await repo.get_user_history(user.id)

        # Most recent should be first
        for i in range(len(history) - 1):
            assert history[i].created_at >= history[i + 1].created_at


class TestVehicleMakeRepository:
    """Test VehicleMake repository operations."""

    @pytest.mark.asyncio
    async def test_create_vehicle_make(self, db_session):
        """Test creating a vehicle make."""
        repo = VehicleMakeRepository(db_session)

        make = await repo.create({
            "id": "tesla",
            "name": "Tesla",
            "country": "USA",
        })

        assert make.id == "tesla"
        assert make.name == "Tesla"

    @pytest.mark.asyncio
    async def test_get_vehicle_make(self, db_session, seeded_db):
        """Test getting a vehicle make by ID."""
        repo = VehicleMakeRepository(db_session)

        make = await repo.get("volkswagen")

        assert make is not None
        assert make.name == "Volkswagen"

    @pytest.mark.asyncio
    async def test_search_vehicle_makes(self, db_session, seeded_db):
        """Test searching vehicle makes."""
        repo = VehicleMakeRepository(db_session)

        results = await repo.search("volks")

        assert len(results) >= 1
        names = [r.name.lower() for r in results]
        assert any("volks" in name for name in names)


class TestVehicleModelRepository:
    """Test VehicleModel repository operations."""

    @pytest.mark.asyncio
    async def test_create_vehicle_model(self, db_session, seeded_db):
        """Test creating a vehicle model."""
        repo = VehicleModelRepository(db_session)

        model = await repo.create({
            "id": "golf-8",
            "name": "Golf 8",
            "make_id": "volkswagen",
            "year_start": 2020,
            "body_types": ["Hatchback", "Wagon"],
        })

        assert model.id == "golf-8"
        assert model.make_id == "volkswagen"

    @pytest.mark.asyncio
    async def test_get_models_by_make(self, db_session, seeded_db):
        """Test getting models by make."""
        # First create some models
        repo = VehicleModelRepository(db_session)

        await repo.create({
            "id": "golf",
            "name": "Golf",
            "make_id": "volkswagen",
            "year_start": 1974,
        })
        await repo.create({
            "id": "passat",
            "name": "Passat",
            "make_id": "volkswagen",
            "year_start": 1973,
        })

        # Get models for VW
        models = await repo.get_by_make("volkswagen")

        assert len(models) >= 2
        model_names = [m.name for m in models]
        assert "Golf" in model_names
        assert "Passat" in model_names

    @pytest.mark.asyncio
    async def test_get_models_by_make_with_year_filter(self, db_session, seeded_db):
        """Test getting models filtered by year."""
        repo = VehicleModelRepository(db_session)

        # Create models with different year ranges
        await repo.create({
            "id": "id3",
            "name": "ID.3",
            "make_id": "volkswagen",
            "year_start": 2020,
        })
        await repo.create({
            "id": "beetle-old",
            "name": "Beetle Classic",
            "make_id": "volkswagen",
            "year_start": 1938,
            "year_end": 2003,
        })

        # Get models available in 2021
        models = await repo.get_by_make("volkswagen", year=2021)

        model_names = [m.name for m in models]
        assert "ID.3" in model_names
        assert "Beetle Classic" not in model_names


class TestDatabaseTransactions:
    """Test database transaction handling."""

    @pytest.mark.asyncio
    async def test_rollback_on_error(self, db_session):
        """Test that errors trigger rollback."""
        repo = DTCCodeRepository(db_session)

        # Create a valid code
        await repo.create({
            "code": "P0400",
            "description_en": "Test code",
            "category": "powertrain",
        })

        # Try to create duplicate (should fail)
        try:
            await repo.create({
                "code": "P0400",  # Duplicate
                "description_en": "Duplicate test code",
                "category": "powertrain",
            })
        except Exception:
            pass

        # Session should still be usable
        dtc = await repo.get_by_code("P0400")
        assert dtc is not None


class TestDatabaseConstraints:
    """Test database constraint enforcement."""

    @pytest.mark.asyncio
    async def test_unique_dtc_code(self, db_session, seeded_db):
        """Test that DTC code must be unique."""
        repo = DTCCodeRepository(db_session)

        # P0101 already exists in seeded data
        with pytest.raises(Exception):
            await repo.create({
                "code": "P0101",  # Duplicate
                "description_en": "Duplicate code",
                "category": "powertrain",
            })

    @pytest.mark.asyncio
    async def test_unique_user_email(self, db_session, seeded_db):
        """Test that user email must be unique."""
        repo = UserRepository(db_session)

        # test@example.com already exists in seeded data
        with pytest.raises(Exception):
            await repo.create({
                "email": "test@example.com",  # Duplicate
                "hashed_password": "hash",
            })

    @pytest.mark.asyncio
    async def test_required_fields(self, db_session):
        """Test that required fields are enforced."""
        repo = DTCCodeRepository(db_session)

        # Missing required description_en
        with pytest.raises(Exception):
            await repo.create({
                "code": "P0500",
                "category": "powertrain",
                # Missing description_en
            })
