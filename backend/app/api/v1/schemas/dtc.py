"""
DTC (Diagnostic Trouble Code) schemas.
"""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from app.core.dtc_codes import normalize_dtc_code

# Python 3.9 compatible string enum
from enum import Enum


class DTCCategory(str, Enum):
    """DTC code categories (Python 3.9 compatible)."""

    POWERTRAIN = "powertrain"  # P codes
    BODY = "body"  # B codes
    CHASSIS = "chassis"  # C codes
    NETWORK = "network"  # U codes

    def __str__(self) -> str:
        return str(self.value)


class DTCSeverity(str, Enum):
    """DTC severity levels (Python 3.9 compatible)."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    def __str__(self) -> str:
        return str(self.value)


class DTCCode(BaseModel):
    """Basic DTC code schema."""

    code: str = Field(..., description="DTC code (e.g., P0101)")
    description_en: str = Field(..., description="English description")
    description_hu: Optional[str] = Field(None, description="Hungarian description")
    category: str = Field(..., description="Category (powertrain, body, chassis, network)")
    is_generic: bool = Field(True, description="Whether this is a generic OBD-II code")

    class Config:
        from_attributes = True


class DTCSearchResult(DTCCode):
    """DTC code search result with additional fields."""

    severity: str = Field("medium", description="Severity level")
    relevance_score: float = Field(0.0, ge=0, le=1, description="Search relevance score")


class DTCCodeDetail(DTCCode):
    """Detailed DTC code information."""

    severity: str = Field("medium", description="Severity level")
    system: Optional[str] = Field(None, description="System/subsystem affected")

    symptoms: List[str] = Field(default_factory=list, description="Common symptoms in Hungarian")
    possible_causes: List[str] = Field(
        default_factory=list, description="Possible causes in Hungarian"
    )
    diagnostic_steps: List[str] = Field(
        default_factory=list, description="Diagnostic steps in Hungarian"
    )

    related_codes: List[str] = Field(default_factory=list, description="Related DTC codes")
    common_vehicles: List[str] = Field(
        default_factory=list, description="Commonly affected vehicles"
    )

    manufacturer_code: Optional[str] = Field(None, description="Manufacturer-specific code")
    freeze_frame_data: Optional[List[str]] = Field(
        None, description="Expected freeze frame parameters"
    )


class DTCCreate(BaseModel):
    """Schema for creating a new DTC code entry.

    ``code`` is validated against the SAE J2012 rule in ``app.core.dtc_codes``,
    the same primitive ``GET /api/v1/dtc/{code}`` uses. Before that, the only
    constraint was ``5 <= len(code) <= 10``, so the write path accepted
    spellings the read path answers with 400 - a row you can insert and then
    cannot open. That is how ``PEACE``, ``PACED``, ``P93AF``, ``UA80E`` and
    ``UA80F`` reached the corpus: hex-shaped English words and manufacturer
    designations whose second character is outside ``0-3``.
    """

    code: str = Field(
        ...,
        min_length=5,
        max_length=10,
        description="DTC code, SAE J2012 format (e.g. P0101, P26B7, B00A0, U0100)",
    )
    description_en: str = Field(..., min_length=5, max_length=500)
    description_hu: Optional[str] = Field(None, max_length=500)
    category: DTCCategory
    severity: DTCSeverity = DTCSeverity.MEDIUM
    is_generic: bool = True
    system: Optional[str] = Field(None, max_length=100)
    symptoms: List[str] = Field(default_factory=list)
    possible_causes: List[str] = Field(default_factory=list)
    diagnostic_steps: List[str] = Field(default_factory=list)
    related_codes: List[str] = Field(default_factory=list)

    @field_validator("code")
    @classmethod
    def validate_code(cls, v: str) -> str:
        """Reject non-DTC spellings and return the canonical upper-case form.

        Args:
            v: The submitted code.

        Returns:
            The canonical upper-case code (``" p0300 "`` -> ``"P0300"``).

        Raises:
            ValueError: If the code is not structurally a DTC.

        Structural rules live in ``app.core.dtc_codes`` (SAE J2012), shared
        with the read path, the request validators and the importer scripts:
        real hex codes such as ``P26B7`` and ``B00A0`` are accepted, while
        ``PEACE`` / ``P8888`` / ``U760E`` (second character outside ``0-3``)
        are not. Deliberately NOT a new regex - the project just consolidated
        ten of those into this one primitive.

        Canonicalising here rather than at the endpoint means every consumer
        (uniqueness check, ORM row, ``Location`` header, log line) sees the one
        spelling the detail endpoint will later accept.
        """
        canonical = normalize_dtc_code(v)
        if canonical is None:
            raise ValueError(
                f"Invalid DTC code format: {v!r}. "
                "Expected SAE J2012 format, e.g. P0101, B1234, C0567, U0100"
            )
        return canonical


class DTCBulkImport(BaseModel):
    """Schema for bulk importing DTC codes.

    Note for admin import flows: because each item is a :class:`DTCCreate`, a
    payload containing even one malformed code is rejected as a whole with 422
    before the endpoint body runs - it does NOT arrive as a per-item entry in
    the endpoint's ``errors`` array. That is the intended trade: a corpus is
    worth failing a batch over, the caller is an authenticated admin, and
    Pydantic names the offending element (``body.codes.7.code``) so the fix is
    mechanical. Re-importing a legacy dump that still carries the historical
    junk codes will now fail loudly instead of silently re-seeding them.
    """

    codes: List[DTCCreate] = Field(..., min_length=1, max_length=1000)
    overwrite_existing: bool = Field(False, description="Overwrite existing codes with same code")
