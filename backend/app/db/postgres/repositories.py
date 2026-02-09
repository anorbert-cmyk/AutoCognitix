"""
Repository pattern implementations for database operations.

This module provides repository classes for database operations following
the repository pattern for clean separation of data access logic.

Performance Optimizations:
- Optimized queries with proper indexing hints
- Batch operations for bulk data access
- Count queries with EXISTS for efficiency
- Cursor-based pagination for large datasets
- Full-text search using PostgreSQL GIN indexes
"""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Generic, TypeVar
from uuid import UUID

from sqlalchemy import and_, func, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.postgres.models import Base, DiagnosisSession, DTCCode, User, VehicleMake, VehicleModel

if TYPE_CHECKING:
    from sqlalchemy.sql.elements import ColumnElement

# Generic type for models
ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository(Generic[ModelType]):
    """
    Base repository with common CRUD operations.

    Provides generic create, read, update, delete operations for SQLAlchemy models.

    Attributes:
        model: The SQLAlchemy model class.
        db: The async database session.
    """

    def __init__(self, model: type[ModelType], db: AsyncSession) -> None:
        """
        Initialize the repository.

        Args:
            model: The SQLAlchemy model class.
            db: The async database session.
        """
        self.model = model
        self.db = db

    async def get(self, id: str | int | UUID) -> ModelType | None:
        """Get a single record by ID."""
        result = await self.db.execute(select(self.model).where(self.model.id == id))  # type: ignore[attr-defined]
        return result.scalar_one_or_none()

    async def get_all(self, skip: int = 0, limit: int = 100) -> list[ModelType]:
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

    async def update(self, id: str | int | UUID, obj_in: dict) -> ModelType | None:
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
    """Repository for User operations with email-based lookups and authentication support."""

    # Account lockout settings
    MAX_FAILED_ATTEMPTS: int = 5
    LOCKOUT_DURATION_MINUTES: int = 30

    def __init__(self, db: AsyncSession) -> None:
        """Initialize the User repository."""
        super().__init__(User, db)

    async def get_by_email(self, email: str) -> User | None:
        """
        Get user by email address.

        Args:
            email: The email address to search for.

        Returns:
            User if found, None otherwise.
        """
        result = await self.db.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()

    async def is_account_locked(self, user: User) -> bool:
        """
        Check if a user account is currently locked due to failed login attempts.

        Args:
            user: The user to check.

        Returns:
            True if account is locked, False otherwise.
        """
        if not user.locked_until:
            return False

        # Check if lockout period has expired
        now = datetime.utcnow()
        if user.locked_until <= now:
            # Lockout expired, reset the counter
            user.locked_until = None
            user.failed_login_attempts = 0
            await self.db.flush()
            return False

        return True

    async def record_failed_login(self, user: User) -> bool:
        """
        Record a failed login attempt and lock account if threshold reached.

        Args:
            user: The user who failed to login.

        Returns:
            True if account is now locked, False otherwise.
        """
        from datetime import timedelta

        user.failed_login_attempts = (user.failed_login_attempts or 0) + 1
        user.last_failed_login = datetime.utcnow()

        # Lock account if max attempts exceeded
        if user.failed_login_attempts >= self.MAX_FAILED_ATTEMPTS:
            user.locked_until = datetime.utcnow() + timedelta(minutes=self.LOCKOUT_DURATION_MINUTES)
            await self.db.flush()
            return True

        await self.db.flush()
        return False

    async def record_successful_login(self, user: User) -> None:
        """
        Record a successful login and reset failed attempt counter.

        Args:
            user: The user who logged in successfully.
        """
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login_at = datetime.utcnow()
        await self.db.flush()

    async def update_password(self, user: User, hashed_password: str) -> None:
        """
        Update user's password.

        Args:
            user: The user to update.
            hashed_password: The new hashed password.
        """
        user.hashed_password = hashed_password
        user.password_reset_token = None
        user.password_reset_expires = None
        await self.db.flush()

    async def set_password_reset_token(self, user: User, token: str) -> None:
        """
        Set password reset token for a user.

        Args:
            user: The user requesting password reset.
            token: The password reset token (JWT).
        """
        from datetime import timedelta

        user.password_reset_token = token
        user.password_reset_expires = datetime.utcnow() + timedelta(hours=1)
        await self.db.flush()

    async def verify_email(self, user: User) -> None:
        """
        Mark user's email as verified.

        Args:
            user: The user whose email was verified.
        """
        user.is_email_verified = True
        user.email_verification_token = None
        await self.db.flush()

    async def set_email_verification_token(self, user: User, token: str) -> None:
        """
        Set email verification token for a user.

        Args:
            user: The user to set the token for.
            token: The verification token.
        """
        user.email_verification_token = token
        await self.db.flush()

    async def get_active_users(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> list[User]:
        """
        Get all active users with pagination.

        Args:
            skip: Number of records to skip.
            limit: Maximum number of records to return.

        Returns:
            List of active User objects.
        """
        result = await self.db.execute(
            select(User)
            .where(User.is_active == True)  # noqa: E712
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def deactivate(self, user: User) -> None:
        """
        Deactivate a user account.

        Args:
            user: The user to deactivate.
        """
        user.is_active = False
        await self.db.flush()

    async def activate(self, user: User) -> None:
        """
        Activate a user account.

        Args:
            user: The user to activate.
        """
        user.is_active = True
        await self.db.flush()


class DTCCodeRepository(BaseRepository[DTCCode]):
    """Repository for DTC (Diagnostic Trouble Code) operations with search capabilities."""

    def __init__(self, db: AsyncSession) -> None:
        """Initialize the DTC code repository."""
        super().__init__(DTCCode, db)

    async def get_by_code(self, code: str) -> DTCCode | None:
        """
        Get DTC by code string.

        Args:
            code: The DTC code to search for (case-insensitive).

        Returns:
            DTCCode if found, None otherwise.
        """
        result = await self.db.execute(select(DTCCode).where(DTCCode.code == code.upper()))
        return result.scalar_one_or_none()

    async def search(
        self,
        query: str,
        category: str | None = None,
        limit: int = 20,
    ) -> list[DTCCode]:
        """
        Search DTC codes by query string.

        Searches in code, English description, and Hungarian description fields.

        Args:
            query: Search query string.
            category: Optional category filter (powertrain, body, chassis, network).
            limit: Maximum number of results to return.

        Returns:
            List of matching DTCCode objects.
        """
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

    async def get_related_codes(self, code: str) -> list[DTCCode]:
        """
        Get DTC codes related to the specified code.

        Args:
            code: The DTC code to find related codes for.

        Returns:
            List of related DTCCode objects.
        """
        dtc = await self.get_by_code(code)
        if not dtc or not dtc.related_codes:
            return []

        result = await self.db.execute(select(DTCCode).where(DTCCode.code.in_(dtc.related_codes)))
        return list(result.scalars().all())


class VehicleMakeRepository(BaseRepository[VehicleMake]):
    """Repository for vehicle make operations."""

    def __init__(self, db: AsyncSession):
        super().__init__(VehicleMake, db)

    async def search(self, query: str) -> list[VehicleMake]:
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
        year: int | None = None,
    ) -> list[VehicleModel]:
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
    ) -> list[DiagnosisSession]:
        """Get diagnosis history for a user."""
        result = await self.db.execute(
            select(DiagnosisSession)
            .where(
                and_(
                    DiagnosisSession.user_id == user_id,
                    DiagnosisSession.is_deleted.is_(False),
                )
            )
            .order_by(DiagnosisSession.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_filtered_history(
        self,
        user_id: UUID,
        vehicle_make: str | None = None,
        vehicle_model: str | None = None,
        vehicle_year: int | None = None,
        dtc_code: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        skip: int = 0,
        limit: int = 10,
    ) -> tuple[list[DiagnosisSession], int]:
        """
        Get filtered diagnosis history with total count.

        Uses optimized queries with proper index utilization.

        Args:
            user_id: User UUID
            vehicle_make: Filter by vehicle make (partial match)
            vehicle_model: Filter by vehicle model (partial match)
            vehicle_year: Filter by exact vehicle year
            dtc_code: Filter by DTC code (checks array contains)
            date_from: Filter by start date
            date_to: Filter by end date
            skip: Pagination offset
            limit: Page size

        Returns:
            Tuple of (items, total_count)
        """
        # Build base query
        conditions: list[ColumnElement[bool]] = [
            DiagnosisSession.user_id == user_id,
            DiagnosisSession.is_deleted.is_(False),
        ]

        if vehicle_make:
            conditions.append(DiagnosisSession.vehicle_make.ilike(f"%{vehicle_make}%"))

        if vehicle_model:
            conditions.append(DiagnosisSession.vehicle_model.ilike(f"%{vehicle_model}%"))

        if vehicle_year:
            conditions.append(DiagnosisSession.vehicle_year == vehicle_year)

        if dtc_code:
            # Use GIN index on dtc_codes array
            conditions.append(DiagnosisSession.dtc_codes.contains([dtc_code.upper()]))

        if date_from:
            conditions.append(DiagnosisSession.created_at >= date_from)

        if date_to:
            conditions.append(DiagnosisSession.created_at <= date_to)

        # Build the query with all conditions
        base_query = select(DiagnosisSession).where(and_(*conditions))

        # Get total count efficiently
        count_query = select(func.count()).select_from(base_query.subquery())
        count_result = await self.db.execute(count_query)
        total = count_result.scalar() or 0

        # Get paginated results
        items_query = (
            base_query.order_by(DiagnosisSession.created_at.desc()).offset(skip).limit(limit)
        )
        items_result = await self.db.execute(items_query)
        items = list(items_result.scalars().all())

        return items, total

    async def soft_delete(
        self,
        diagnosis_id: UUID,
        user_id: UUID | None = None,
    ) -> bool:
        """
        Soft delete a diagnosis session.

        Args:
            diagnosis_id: UUID of the diagnosis to delete
            user_id: Optional user ID to verify ownership

        Returns:
            True if deleted, False if not found
        """
        conditions = [DiagnosisSession.id == diagnosis_id]
        if user_id:
            conditions.append(DiagnosisSession.user_id == user_id)

        result = await self.db.execute(select(DiagnosisSession).where(and_(*conditions)))
        session = result.scalar_one_or_none()

        if not session:
            return False

        session.is_deleted = True
        session.deleted_at = datetime.utcnow()
        await self.db.flush()
        return True

    async def get_user_stats(self, user_id: UUID) -> dict[str, Any]:
        """
        Get diagnosis statistics for a user.

        Uses efficient aggregate queries.

        Args:
            user_id: User UUID

        Returns:
            Statistics dictionary with total, average, top vehicles, and monthly trends
        """
        # Total diagnoses and average confidence
        stats_query = select(
            func.count(DiagnosisSession.id).label("total"),
            func.avg(DiagnosisSession.confidence_score).label("avg_confidence"),
        ).where(
            and_(
                DiagnosisSession.user_id == user_id,
                DiagnosisSession.is_deleted.is_(False),
            )
        )
        stats_result = await self.db.execute(stats_query)
        stats_row = stats_result.one()

        # Most diagnosed vehicles
        vehicles_query = (
            select(
                DiagnosisSession.vehicle_make,
                DiagnosisSession.vehicle_model,
                func.count(DiagnosisSession.id).label("count"),
            )
            .where(
                and_(
                    DiagnosisSession.user_id == user_id,
                    DiagnosisSession.is_deleted.is_(False),
                )
            )
            .group_by(DiagnosisSession.vehicle_make, DiagnosisSession.vehicle_model)
            .order_by(text("count DESC"))
            .limit(5)
        )
        vehicles_result = await self.db.execute(vehicles_query)
        vehicles = [
            {"make": row.vehicle_make, "model": row.vehicle_model, "count": row.count}
            for row in vehicles_result
        ]

        # Monthly diagnosis counts (last 12 months)
        monthly_query = (
            select(
                func.to_char(DiagnosisSession.created_at, "YYYY-MM").label("month"),
                func.count(DiagnosisSession.id).label("count"),
            )
            .where(
                and_(
                    DiagnosisSession.user_id == user_id,
                    DiagnosisSession.is_deleted.is_(False),
                    DiagnosisSession.created_at >= func.now() - text("interval '12 months'"),
                )
            )
            .group_by(text("month"))
            .order_by(text("month DESC"))
        )
        monthly_result = await self.db.execute(monthly_query)
        monthly = [{"month": row.month, "count": row.count} for row in monthly_result]

        return {
            "total_diagnoses": stats_row.total or 0,
            "avg_confidence": float(stats_row.avg_confidence or 0),
            "most_diagnosed_vehicles": vehicles,
            "diagnoses_by_month": monthly,
        }

    async def get_dtc_frequency(
        self,
        user_id: UUID | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Get most common DTC codes from diagnosis history.

        Uses PostgreSQL array unnest for efficient counting.

        Args:
            user_id: Optional user ID to filter by
            limit: Maximum number of codes to return

        Returns:
            List of {code, count} dictionaries
        """
        # Use unnest to expand the dtc_codes array and count occurrences
        conditions: list[ColumnElement[bool]] = [DiagnosisSession.is_deleted.is_(False)]
        if user_id:
            conditions.append(DiagnosisSession.user_id == user_id)

        query = text("""
            SELECT code, COUNT(*) as count
            FROM diagnosis_sessions, unnest(dtc_codes) as code
            WHERE user_id = :user_id AND is_deleted = false
            GROUP BY code
            ORDER BY count DESC
            LIMIT :limit
        """)

        if user_id:
            result = await self.db.execute(query, {"user_id": str(user_id), "limit": limit})
        else:
            # Global frequency (without user filter)
            query = text("""
                SELECT code, COUNT(*) as count
                FROM diagnosis_sessions, unnest(dtc_codes) as code
                WHERE is_deleted = false
                GROUP BY code
                ORDER BY count DESC
                LIMIT :limit
            """)
            result = await self.db.execute(query, {"limit": limit})

        return [{"code": row.code, "count": row.count} for row in result]


class DTCCodeBatchRepository:
    """
    Batch operations repository for DTC codes.

    Optimized for bulk operations and batch lookups.
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def get_batch(self, codes: list[str]) -> dict[str, DTCCode]:
        """
        Get multiple DTC codes in a single query.

        Args:
            codes: List of DTC codes to fetch

        Returns:
            Dictionary mapping code to DTCCode object
        """
        if not codes:
            return {}

        upper_codes = [c.upper() for c in codes]
        result = await self.db.execute(select(DTCCode).where(DTCCode.code.in_(upper_codes)))
        dtcs = result.scalars().all()
        return {dtc.code: dtc for dtc in dtcs}

    async def search_fulltext(
        self,
        query: str,
        limit: int = 20,
    ) -> list[DTCCode]:
        """
        Full-text search using PostgreSQL FTS index.

        Uses the ix_dtc_codes_description_fts GIN index.

        Args:
            query: Search query (will be converted to tsquery)
            limit: Maximum results

        Returns:
            List of matching DTCCode objects
        """
        # Convert query to tsquery format (simple configuration for multi-language)
        fts_query = text("""
            SELECT * FROM dtc_codes
            WHERE to_tsvector('simple', COALESCE(description_en, '') || ' ' || COALESCE(description_hu, ''))
                  @@ plainto_tsquery('simple', :query)
            ORDER BY ts_rank(
                to_tsvector('simple', COALESCE(description_en, '') || ' ' || COALESCE(description_hu, '')),
                plainto_tsquery('simple', :query)
            ) DESC
            LIMIT :limit
        """)
        result = await self.db.execute(fts_query, {"query": query, "limit": limit})
        return list(result.scalars().all())

    async def get_by_symptoms(
        self,
        symptoms: list[str],
        limit: int = 20,
    ) -> list[DTCCode]:
        """
        Find DTC codes that match given symptoms.

        Uses the GIN index on symptoms array.

        Args:
            symptoms: List of symptom strings to match
            limit: Maximum results

        Returns:
            List of matching DTCCode objects
        """
        if not symptoms:
            return []

        # Build OR conditions for symptom overlap
        conditions = []
        for symptom in symptoms:
            conditions.append(DTCCode.symptoms.contains([symptom]))

        result = await self.db.execute(select(DTCCode).where(or_(*conditions)).limit(limit))
        return list(result.scalars().all())
