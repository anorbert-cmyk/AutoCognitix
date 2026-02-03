"""
Repository pattern implementations for database operations.
"""

from typing import Generic, List, Optional, Type, TypeVar
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.postgres.models import Base, DTCCode, DiagnosisSession, User, VehicleMake, VehicleModel

# Generic type for models
ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository(Generic[ModelType]):
    """Base repository with common CRUD operations."""

    def __init__(self, model: Type[ModelType], db: AsyncSession):
        self.model = model
        self.db = db

    async def get(self, id: str | int | UUID) -> Optional[ModelType]:
        """Get a single record by ID."""
        result = await self.db.execute(select(self.model).where(self.model.id == id))
        return result.scalar_one_or_none()

    async def get_all(self, skip: int = 0, limit: int = 100) -> List[ModelType]:
        """Get all records with pagination."""
        result = await self.db.execute(select(self.model).offset(skip).limit(limit))
        return list(result.scalars().all())

    async def create(self, obj_in: dict) -> ModelType:
        """Create a new record."""
        db_obj = self.model(**obj_in)
        self.db.add(db_obj)
        await self.db.flush()
        await self.db.refresh(db_obj)
        return db_obj

    async def update(self, id: str | int | UUID, obj_in: dict) -> Optional[ModelType]:
        """Update an existing record."""
        db_obj = await self.get(id)
        if db_obj:
            for key, value in obj_in.items():
                setattr(db_obj, key, value)
            await self.db.flush()
            await self.db.refresh(db_obj)
        return db_obj

    async def delete(self, id: str | int | UUID) -> bool:
        """Delete a record."""
        db_obj = await self.get(id)
        if db_obj:
            await self.db.delete(db_obj)
            await self.db.flush()
            return True
        return False


class UserRepository(BaseRepository[User]):
    """Repository for User operations."""

    def __init__(self, db: AsyncSession):
        super().__init__(User, db)

    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        result = await self.db.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()


class DTCCodeRepository(BaseRepository[DTCCode]):
    """Repository for DTC code operations."""

    def __init__(self, db: AsyncSession):
        super().__init__(DTCCode, db)

    async def get_by_code(self, code: str) -> Optional[DTCCode]:
        """Get DTC by code string."""
        result = await self.db.execute(select(DTCCode).where(DTCCode.code == code.upper()))
        return result.scalar_one_or_none()

    async def search(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 20,
    ) -> List[DTCCode]:
        """Search DTC codes by query string."""
        stmt = select(DTCCode).where(
            (DTCCode.code.ilike(f"%{query}%"))
            | (DTCCode.description_en.ilike(f"%{query}%"))
            | (DTCCode.description_hu.ilike(f"%{query}%"))
        )

        if category:
            stmt = stmt.where(DTCCode.category == category)

        stmt = stmt.limit(limit)
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def get_related_codes(self, code: str) -> List[DTCCode]:
        """Get related DTC codes."""
        dtc = await self.get_by_code(code)
        if not dtc or not dtc.related_codes:
            return []

        result = await self.db.execute(
            select(DTCCode).where(DTCCode.code.in_(dtc.related_codes))
        )
        return list(result.scalars().all())


class VehicleMakeRepository(BaseRepository[VehicleMake]):
    """Repository for vehicle make operations."""

    def __init__(self, db: AsyncSession):
        super().__init__(VehicleMake, db)

    async def search(self, query: str) -> List[VehicleMake]:
        """Search makes by name."""
        result = await self.db.execute(
            select(VehicleMake).where(VehicleMake.name.ilike(f"%{query}%"))
        )
        return list(result.scalars().all())


class VehicleModelRepository(BaseRepository[VehicleModel]):
    """Repository for vehicle model operations."""

    def __init__(self, db: AsyncSession):
        super().__init__(VehicleModel, db)

    async def get_by_make(
        self,
        make_id: str,
        year: Optional[int] = None,
    ) -> List[VehicleModel]:
        """Get models by make ID, optionally filtered by year."""
        stmt = select(VehicleModel).where(VehicleModel.make_id == make_id)

        if year:
            stmt = stmt.where(
                (VehicleModel.year_start <= year)
                & ((VehicleModel.year_end.is_(None)) | (VehicleModel.year_end >= year))
            )

        result = await self.db.execute(stmt)
        return list(result.scalars().all())


class DiagnosisSessionRepository(BaseRepository[DiagnosisSession]):
    """Repository for diagnosis session operations."""

    def __init__(self, db: AsyncSession):
        super().__init__(DiagnosisSession, db)

    async def get_user_history(
        self,
        user_id: UUID,
        skip: int = 0,
        limit: int = 10,
    ) -> List[DiagnosisSession]:
        """Get diagnosis history for a user."""
        result = await self.db.execute(
            select(DiagnosisSession)
            .where(DiagnosisSession.user_id == user_id)
            .order_by(DiagnosisSession.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())
