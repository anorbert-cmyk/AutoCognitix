"""
SQLAlchemy models for PostgreSQL database.
"""

from datetime import datetime
from typing import List, Optional
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all models."""

    pass


class User(Base):
    """User model for authentication."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(100))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False)
    role: Mapped[str] = mapped_column(String(50), default="user")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    diagnosis_sessions = relationship("DiagnosisSession", back_populates="user")


class VehicleMake(Base):
    """Vehicle manufacturer/make model."""

    __tablename__ = "vehicle_makes"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    country: Mapped[Optional[str]] = mapped_column(String(100))
    logo_url: Mapped[Optional[str]] = mapped_column(String(500))

    # Relationships
    models = relationship("VehicleModel", back_populates="make")


class VehicleModel(Base):
    """Vehicle model."""

    __tablename__ = "vehicle_models"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    make_id: Mapped[str] = mapped_column(String(50), ForeignKey("vehicle_makes.id"), nullable=False)
    year_start: Mapped[int] = mapped_column(Integer, nullable=False)
    year_end: Mapped[Optional[int]] = mapped_column(Integer)
    body_types: Mapped[List[str]] = mapped_column(ARRAY(String), default=[])
    engine_codes: Mapped[List[str]] = mapped_column(ARRAY(String), default=[])
    platform: Mapped[Optional[str]] = mapped_column(String(50))

    # Relationships
    make = relationship("VehicleMake", back_populates="models")


class DTCCode(Base):
    """Diagnostic Trouble Code model."""

    __tablename__ = "dtc_codes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(10), unique=True, index=True, nullable=False)
    description_en: Mapped[str] = mapped_column(Text, nullable=False)
    description_hu: Mapped[Optional[str]] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(20), nullable=False)  # powertrain, body, chassis, network
    severity: Mapped[str] = mapped_column(String(20), default="medium")  # low, medium, high, critical
    is_generic: Mapped[bool] = mapped_column(Boolean, default=True)
    system: Mapped[Optional[str]] = mapped_column(String(100))

    # Arrays for detailed info
    symptoms: Mapped[List[str]] = mapped_column(ARRAY(String), default=[])
    possible_causes: Mapped[List[str]] = mapped_column(ARRAY(String), default=[])
    diagnostic_steps: Mapped[List[str]] = mapped_column(ARRAY(String), default=[])
    related_codes: Mapped[List[str]] = mapped_column(ARRAY(String), default=[])

    # Manufacturer-specific info
    manufacturer_code: Mapped[Optional[str]] = mapped_column(String(50))
    applicable_makes: Mapped[List[str]] = mapped_column(ARRAY(String), default=[])

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
    symptoms: Mapped[List[str]] = mapped_column(ARRAY(String), default=[])
    causes: Mapped[List[str]] = mapped_column(ARRAY(String), default=[])
    solutions: Mapped[List[str]] = mapped_column(ARRAY(String), default=[])
    related_dtc_codes: Mapped[List[str]] = mapped_column(ARRAY(String), default=[])

    # Vehicle applicability
    applicable_makes: Mapped[List[str]] = mapped_column(ARRAY(String), default=[])
    applicable_models: Mapped[List[str]] = mapped_column(ARRAY(String), default=[])
    year_start: Mapped[Optional[int]] = mapped_column(Integer)
    year_end: Mapped[Optional[int]] = mapped_column(Integer)

    # Metadata
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    source_type: Mapped[Optional[str]] = mapped_column(String(50))  # tsb, forum, database
    source_url: Mapped[Optional[str]] = mapped_column(String(500))

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class DiagnosisSession(Base):
    """Diagnosis session/history model."""

    __tablename__ = "diagnosis_sessions"

    id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[Optional[str]] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"))

    # Vehicle info
    vehicle_make: Mapped[str] = mapped_column(String(100), nullable=False)
    vehicle_model: Mapped[str] = mapped_column(String(100), nullable=False)
    vehicle_year: Mapped[int] = mapped_column(Integer, nullable=False)
    vehicle_vin: Mapped[Optional[str]] = mapped_column(String(17))

    # Input data
    dtc_codes: Mapped[List[str]] = mapped_column(ARRAY(String), nullable=False)
    symptoms_text: Mapped[str] = mapped_column(Text, nullable=False)
    additional_context: Mapped[Optional[str]] = mapped_column(Text)

    # Results (stored as JSON)
    diagnosis_result: Mapped[dict] = mapped_column(JSONB, nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    user = relationship("User", back_populates="diagnosis_sessions")
