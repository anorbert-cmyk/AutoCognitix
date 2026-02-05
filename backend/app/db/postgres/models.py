"""
SQLAlchemy models for PostgreSQL database.
"""

from datetime import date, datetime
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all models."""

    pass


class User(Base):
    """User model for authentication with account lockout support."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str | None] = mapped_column(String(100))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False)
    role: Mapped[str] = mapped_column(String(50), default="user")  # user, mechanic, admin

    # Account lockout fields
    failed_login_attempts: Mapped[int] = mapped_column(Integer, default=0)
    locked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_failed_login: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Password reset fields
    password_reset_token: Mapped[str | None] = mapped_column(String(255))
    password_reset_expires: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Email verification fields
    is_email_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    email_verification_token: Mapped[str | None] = mapped_column(String(255))

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Relationships
    diagnosis_sessions = relationship("DiagnosisSession", back_populates="user")


class VehicleMake(Base):
    """Vehicle manufacturer/make model."""

    __tablename__ = "vehicle_makes"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    country: Mapped[str | None] = mapped_column(String(100))
    logo_url: Mapped[str | None] = mapped_column(String(500))

    # NHTSA API reference
    nhtsa_make_id: Mapped[int | None] = mapped_column(Integer, index=True)

    # Relationships
    models = relationship("VehicleModel", back_populates="make")
    dtc_frequencies = relationship("VehicleDTCFrequency", back_populates="make", cascade="all, delete-orphan")
    tsb_items = relationship("VehicleTSB", back_populates="make", cascade="all, delete-orphan")


class VehicleModel(Base):
    """Vehicle model."""

    __tablename__ = "vehicle_models"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    make_id: Mapped[str] = mapped_column(String(50), ForeignKey("vehicle_makes.id"), nullable=False)
    year_start: Mapped[int] = mapped_column(Integer, nullable=False)
    year_end: Mapped[int | None] = mapped_column(Integer)
    body_types: Mapped[list[str]] = mapped_column(ARRAY(String), default=[])
    engine_codes: Mapped[list[str]] = mapped_column(ARRAY(String), default=[])
    platform: Mapped[str | None] = mapped_column(String(50))
    platform_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("vehicle_platforms.id", ondelete="SET NULL"))

    # Relationships
    make = relationship("VehicleMake", back_populates="models")
    platform_ref = relationship("VehiclePlatform", back_populates="models")
    model_engines = relationship("VehicleModelEngine", back_populates="model", cascade="all, delete-orphan")
    dtc_frequencies = relationship("VehicleDTCFrequency", back_populates="model", cascade="all, delete-orphan")


class DTCCode(Base):
    """Diagnostic Trouble Code model."""

    __tablename__ = "dtc_codes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(10), unique=True, index=True, nullable=False)
    description_en: Mapped[str] = mapped_column(Text, nullable=False)
    description_hu: Mapped[str | None] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(20), nullable=False)  # powertrain, body, chassis, network
    severity: Mapped[str] = mapped_column(String(20), default="medium")  # low, medium, high, critical
    is_generic: Mapped[bool] = mapped_column(Boolean, default=True)
    system: Mapped[str | None] = mapped_column(String(100))

    # Arrays for detailed info
    symptoms: Mapped[list[str]] = mapped_column(ARRAY(String), default=[])
    possible_causes: Mapped[list[str]] = mapped_column(ARRAY(String), default=[])
    diagnostic_steps: Mapped[list[str]] = mapped_column(ARRAY(String), default=[])
    related_codes: Mapped[list[str]] = mapped_column(ARRAY(String), default=[])

    # Manufacturer-specific info
    manufacturer_code: Mapped[str | None] = mapped_column(String(50))
    applicable_makes: Mapped[list[str]] = mapped_column(ARRAY(String), default=[])

    # Data sources tracking (e.g., "generic", "mytrile", "klavkarr")
    sources: Mapped[list[str]] = mapped_column(ARRAY(String), default=[])

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class KnownIssue(Base):
    """Known issue/common problem model."""

    __tablename__ = "known_issues"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)

    # Problem details
    symptoms: Mapped[list[str]] = mapped_column(ARRAY(String), default=[])
    causes: Mapped[list[str]] = mapped_column(ARRAY(String), default=[])
    solutions: Mapped[list[str]] = mapped_column(ARRAY(String), default=[])
    related_dtc_codes: Mapped[list[str]] = mapped_column(ARRAY(String), default=[])

    # Vehicle applicability
    applicable_makes: Mapped[list[str]] = mapped_column(ARRAY(String), default=[])
    applicable_models: Mapped[list[str]] = mapped_column(ARRAY(String), default=[])
    year_start: Mapped[int | None] = mapped_column(Integer)
    year_end: Mapped[int | None] = mapped_column(Integer)

    # Metadata
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    source_type: Mapped[str | None] = mapped_column(String(50))  # tsb, forum, database
    source_url: Mapped[str | None] = mapped_column(String(500))

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class DiagnosisSession(Base):
    """Diagnosis session/history model."""

    __tablename__ = "diagnosis_sessions"

    id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[str | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), index=True)

    # Vehicle info
    vehicle_make: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    vehicle_model: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    vehicle_year: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    vehicle_vin: Mapped[str | None] = mapped_column(String(17), index=True)

    # Input data
    dtc_codes: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False)
    symptoms_text: Mapped[str] = mapped_column(Text, nullable=False)
    additional_context: Mapped[str | None] = mapped_column(Text)

    # Results (stored as JSON)
    diagnosis_result: Mapped[dict] = mapped_column(JSONB, nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Soft delete
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Relationships
    user = relationship("User", back_populates="diagnosis_sessions")


class VehicleEngine(Base):
    """Vehicle engine specifications model."""

    __tablename__ = "vehicle_engines"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(30), unique=True, index=True, nullable=False)
    name: Mapped[str | None] = mapped_column(String(200))

    # Engine specifications
    displacement_cc: Mapped[int | None] = mapped_column(Integer)  # Displacement in cubic centimeters
    displacement_l: Mapped[float | None] = mapped_column(Float)  # Displacement in liters
    cylinders: Mapped[int | None] = mapped_column(Integer)
    configuration: Mapped[str | None] = mapped_column(String(30))  # inline, v, boxer, rotary
    fuel_type: Mapped[str] = mapped_column(String(30), index=True, nullable=False)  # gasoline, diesel, hybrid, electric, lpg, cng
    aspiration: Mapped[str | None] = mapped_column(String(30))  # naturally_aspirated, turbo, supercharged, twin_turbo

    # Power output
    power_hp: Mapped[int | None] = mapped_column(Integer)
    power_kw: Mapped[int | None] = mapped_column(Integer)
    torque_nm: Mapped[int | None] = mapped_column(Integer)

    # Additional specs
    valves_per_cylinder: Mapped[int | None] = mapped_column(Integer)
    compression_ratio: Mapped[str | None] = mapped_column(String(20))
    bore_mm: Mapped[float | None] = mapped_column(Float)
    stroke_mm: Mapped[float | None] = mapped_column(Float)

    # Manufacturer info
    manufacturer: Mapped[str | None] = mapped_column(String(100))
    family: Mapped[str | None] = mapped_column(String(100))  # Engine family (e.g., EA888, N54, M54)

    # Production years
    year_start: Mapped[int | None] = mapped_column(Integer)
    year_end: Mapped[int | None] = mapped_column(Integer)

    # Applicable makes (for filtering)
    applicable_makes: Mapped[list[str]] = mapped_column(ARRAY(String), default=[])

    # Metadata
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    model_engines = relationship("VehicleModelEngine", back_populates="engine", cascade="all, delete-orphan")


class VehiclePlatform(Base):
    """Vehicle platform (shared across makes) model."""

    __tablename__ = "vehicle_platforms"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(50), unique=True, index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    manufacturer: Mapped[str | None] = mapped_column(String(100))  # Platform owner

    # Shared across makes
    makes: Mapped[list[str]] = mapped_column(ARRAY(String), default=[], nullable=False)

    # Production years
    year_start: Mapped[int | None] = mapped_column(Integer)
    year_end: Mapped[int | None] = mapped_column(Integer)

    # Platform details
    segment: Mapped[str | None] = mapped_column(String(50))  # A, B, C, D, E, F segments
    body_types: Mapped[list[str]] = mapped_column(ARRAY(String), default=[])
    drivetrain_options: Mapped[list[str]] = mapped_column(ARRAY(String), default=[])  # FWD, RWD, AWD

    # Compatible engine codes
    compatible_engines: Mapped[list[str]] = mapped_column(ARRAY(String), default=[])

    # Metadata
    description: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    models = relationship("VehicleModel", back_populates="platform_ref")


class VehicleModelEngine(Base):
    """Many-to-many relationship between vehicle models and engines."""

    __tablename__ = "vehicle_model_engines"
    __table_args__ = (
        UniqueConstraint('model_id', 'engine_id', 'year_start', name='uq_model_engine_year'),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[str] = mapped_column(String(50), ForeignKey("vehicle_models.id", ondelete="CASCADE"), nullable=False)
    engine_id: Mapped[int] = mapped_column(Integer, ForeignKey("vehicle_engines.id", ondelete="CASCADE"), nullable=False)

    # Production years for this combination
    year_start: Mapped[int | None] = mapped_column(Integer)
    year_end: Mapped[int | None] = mapped_column(Integer)

    # Variant info
    variant_name: Mapped[str | None] = mapped_column(String(100))  # e.g., "2.0 TSI 190HP"
    is_base: Mapped[bool] = mapped_column(Boolean, default=False)  # Is this the base engine option

    # Relationships
    model = relationship("VehicleModel", back_populates="model_engines")
    engine = relationship("VehicleEngine", back_populates="model_engines")


class VehicleDTCFrequency(Base):
    """Which DTCs are common for which vehicles."""

    __tablename__ = "vehicle_dtc_frequency"
    __table_args__ = (
        CheckConstraint(
            'make_id IS NOT NULL OR model_id IS NOT NULL OR engine_code IS NOT NULL',
            name='ck_vehicle_dtc_frequency_vehicle_ref'
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dtc_code: Mapped[str] = mapped_column(String(10), ForeignKey("dtc_codes.code"), index=True, nullable=False)

    # Vehicle reference (can be make-level, model-level, or engine-level)
    make_id: Mapped[str | None] = mapped_column(String(50), ForeignKey("vehicle_makes.id", ondelete="CASCADE"), index=True)
    model_id: Mapped[str | None] = mapped_column(String(50), ForeignKey("vehicle_models.id", ondelete="CASCADE"), index=True)
    engine_code: Mapped[str | None] = mapped_column(String(30), index=True)

    # Year range
    year_start: Mapped[int | None] = mapped_column(Integer)
    year_end: Mapped[int | None] = mapped_column(Integer)

    # Frequency data
    frequency: Mapped[str] = mapped_column(String(30), default='common', nullable=False)  # rare, uncommon, common, very_common
    occurrence_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)  # Number of reported occurrences
    confidence: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)  # 0.0-1.0

    # Source info
    source: Mapped[str | None] = mapped_column(String(50))  # nhtsa, tsb, forum, user_reports
    source_url: Mapped[str | None] = mapped_column(String(500))

    # Common symptoms/issues when this DTC appears for this vehicle
    common_symptoms: Mapped[list[str]] = mapped_column(ARRAY(String), default=[])
    common_causes: Mapped[list[str]] = mapped_column(ARRAY(String), default=[])
    known_fixes: Mapped[list[str]] = mapped_column(ARRAY(String), default=[])

    # Metadata
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    make = relationship("VehicleMake", back_populates="dtc_frequencies")
    model = relationship("VehicleModel", back_populates="dtc_frequencies")


class VehicleTSB(Base):
    """Technical Service Bulletin model."""

    __tablename__ = "vehicle_tsb"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    bulletin_number: Mapped[str] = mapped_column(String(50), unique=True, index=True, nullable=False)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)

    # Vehicle applicability
    make_id: Mapped[str | None] = mapped_column(String(50), ForeignKey("vehicle_makes.id", ondelete="CASCADE"), index=True)
    applicable_models: Mapped[list[str]] = mapped_column(ARRAY(String), default=[])
    year_start: Mapped[int | None] = mapped_column(Integer)
    year_end: Mapped[int | None] = mapped_column(Integer)

    # TSB details
    issue_date: Mapped[date | None] = mapped_column(Date)
    component: Mapped[str | None] = mapped_column(String(200))
    related_dtc_codes: Mapped[list[str]] = mapped_column(ARRAY(String), default=[])

    # Source
    source: Mapped[str | None] = mapped_column(String(100))
    source_url: Mapped[str | None] = mapped_column(String(500))

    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    make = relationship("VehicleMake", back_populates="tsb_items")


# =============================================================================
# NHTSA Data Models (from migration 003)
# =============================================================================


class VehicleRecall(Base):
    """NHTSA vehicle recall record."""

    __tablename__ = "vehicle_recalls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    campaign_number: Mapped[str] = mapped_column(String(20), unique=True, index=True, nullable=False)
    nhtsa_id: Mapped[str | None] = mapped_column(String(50))
    manufacturer: Mapped[str] = mapped_column(String(100), nullable=False)
    make: Mapped[str] = mapped_column(String(50), index=True, nullable=False)
    model: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    model_year: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    recall_date: Mapped[date | None] = mapped_column(Date)
    component: Mapped[str | None] = mapped_column(String(500))
    summary: Mapped[str | None] = mapped_column(Text)
    consequence: Mapped[str | None] = mapped_column(Text)
    remedy: Mapped[str | None] = mapped_column(Text)
    notes: Mapped[str | None] = mapped_column(Text)

    # Extracted DTC codes from text analysis
    extracted_dtc_codes: Mapped[list[str]] = mapped_column(ARRAY(String), default=[])

    # Raw API response for reference
    raw_response: Mapped[dict | None] = mapped_column(JSONB)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    dtc_correlations = relationship("DTCRecallCorrelation", back_populates="recall", cascade="all, delete-orphan")


class VehicleComplaint(Base):
    """NHTSA vehicle complaint record."""

    __tablename__ = "vehicle_complaints"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    odi_number: Mapped[str] = mapped_column(String(20), unique=True, index=True, nullable=False)
    manufacturer: Mapped[str] = mapped_column(String(100), nullable=False)
    make: Mapped[str] = mapped_column(String(50), index=True, nullable=False)
    model: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    model_year: Mapped[int] = mapped_column(Integer, index=True, nullable=False)

    # Incident details
    crash: Mapped[bool] = mapped_column(Boolean, default=False)
    fire: Mapped[bool] = mapped_column(Boolean, default=False)
    injuries: Mapped[int] = mapped_column(Integer, default=0)
    deaths: Mapped[int] = mapped_column(Integer, default=0)

    # Dates
    complaint_date: Mapped[date | None] = mapped_column(Date)
    date_of_incident: Mapped[date | None] = mapped_column(Date)

    # Details
    components: Mapped[str | None] = mapped_column(String(500))
    summary: Mapped[str | None] = mapped_column(Text)

    # Extracted DTC codes from text analysis
    extracted_dtc_codes: Mapped[list[str]] = mapped_column(ARRAY(String), default=[])

    # Raw API response for reference
    raw_response: Mapped[dict | None] = mapped_column(JSONB)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    dtc_correlations = relationship("DTCComplaintCorrelation", back_populates="complaint", cascade="all, delete-orphan")


class DTCRecallCorrelation(Base):
    """Link between DTC codes and recalls."""

    __tablename__ = "dtc_recall_correlations"
    __table_args__ = (
        UniqueConstraint("dtc_code", "recall_id", name="uq_dtc_recall_correlation"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dtc_code: Mapped[str] = mapped_column(String(10), ForeignKey("dtc_codes.code"), index=True, nullable=False)
    recall_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("vehicle_recalls.id", ondelete="CASCADE"), index=True, nullable=False
    )
    confidence: Mapped[float] = mapped_column(Float, default=1.0)  # 1.0 = explicit, 0.5 = inferred
    extraction_method: Mapped[str | None] = mapped_column(String(50))  # explicit, component_match, symptom_match
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    recall = relationship("VehicleRecall", back_populates="dtc_correlations")


class DTCComplaintCorrelation(Base):
    """Link between DTC codes and complaints."""

    __tablename__ = "dtc_complaint_correlations"
    __table_args__ = (
        UniqueConstraint("dtc_code", "complaint_id", name="uq_dtc_complaint_correlation"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dtc_code: Mapped[str] = mapped_column(String(10), ForeignKey("dtc_codes.code"), index=True, nullable=False)
    complaint_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("vehicle_complaints.id", ondelete="CASCADE"), index=True, nullable=False
    )
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    extraction_method: Mapped[str | None] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    complaint = relationship("VehicleComplaint", back_populates="dtc_correlations")


class NHTSASyncLog(Base):
    """Track NHTSA data sync progress for incremental updates."""

    __tablename__ = "nhtsa_sync_log"
    __table_args__ = (
        UniqueConstraint("make", "model", "model_year", "data_type", name="uq_nhtsa_sync_log"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    make: Mapped[str] = mapped_column(String(50), index=True, nullable=False)
    model: Mapped[str | None] = mapped_column(String(100))  # NULL = all models
    model_year: Mapped[int] = mapped_column(Integer, nullable=False)
    data_type: Mapped[str] = mapped_column(String(20), nullable=False)  # 'recalls' or 'complaints'
    records_synced: Mapped[int] = mapped_column(Integer, default=0)
    last_synced_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    sync_status: Mapped[str] = mapped_column(String(20), default="completed")  # completed, partial, failed
    error_message: Mapped[str | None] = mapped_column(Text)


class NHTSAVehicleSyncTracking(Base):
    """Track NHTSA vehicle data sync progress for makes, models, recalls, complaints."""

    __tablename__ = "nhtsa_vehicle_sync_tracking"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # What was synced
    sync_type: Mapped[str] = mapped_column(String(30), index=True, nullable=False)  # 'makes', 'models', 'recalls', 'complaints'
    make_name: Mapped[str | None] = mapped_column(String(100), index=True)  # NULL for 'makes' sync type
    model_name: Mapped[str | None] = mapped_column(String(100))  # For model-specific syncs
    year_start: Mapped[int | None] = mapped_column(Integer)
    year_end: Mapped[int | None] = mapped_column(Integer)

    # Sync results
    records_fetched: Mapped[int] = mapped_column(Integer, default=0)
    records_saved: Mapped[int] = mapped_column(Integer, default=0)
    records_skipped: Mapped[int] = mapped_column(Integer, default=0)
    dtc_codes_extracted: Mapped[int] = mapped_column(Integer, default=0)

    # Status
    status: Mapped[str] = mapped_column(String(20), default="completed", index=True)  # 'in_progress', 'completed', 'failed', 'partial'
    error_message: Mapped[str | None] = mapped_column(Text)
    error_count: Mapped[int] = mapped_column(Integer, default=0)

    # API stats
    api_requests: Mapped[int] = mapped_column(Integer, default=0)
    api_errors: Mapped[int] = mapped_column(Integer, default=0)
    elapsed_seconds: Mapped[float | None] = mapped_column(Float)

    # Timestamps
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

    # Extra data (stored as JSONB)
    sync_metadata: Mapped[dict | None] = mapped_column(JSONB)
