"""
DTC (Diagnostic Trouble Code) schemas.
"""

from __future__ import annotations

import sys
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

# Python 3.11+ has StrEnum, provide compatibility for earlier versions
if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    class StrEnum(str, Enum):
        """String enum for Python < 3.11 compatibility."""
        pass


class DTCCategory(StrEnum):
    """DTC code categories."""

    POWERTRAIN = "powertrain"  # P codes
    BODY = "body"  # B codes
    CHASSIS = "chassis"  # C codes
    NETWORK = "network"  # U codes


class DTCSeverity(StrEnum):
    """DTC severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


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
    relevance_score: Optional[float] = Field(None, ge=0, le=1, description="Search relevance score")


class DTCCodeDetail(DTCCode):
    """Detailed DTC code information."""

    severity: str = Field("medium", description="Severity level")
    system: Optional[str] = Field(None, description="System/subsystem affected")

    symptoms: List[str] = Field(default_factory=list, description="Common symptoms in Hungarian")
    possible_causes: List[str] = Field(default_factory=list, description="Possible causes in Hungarian")
    diagnostic_steps: List[str] = Field(default_factory=list, description="Diagnostic steps in Hungarian")

    related_codes: List[str] = Field(default_factory=list, description="Related DTC codes")
    common_vehicles: List[str] = Field(default_factory=list, description="Commonly affected vehicles")

    manufacturer_code: Optional[str] = Field(None, description="Manufacturer-specific code")
    freeze_frame_data: Optional[List[str]] = Field(None, description="Expected freeze frame parameters")


class DTCCreate(BaseModel):
    """Schema for creating a new DTC code entry."""

    code: str = Field(..., min_length=5, max_length=10, description="DTC code")
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


class DTCBulkImport(BaseModel):
    """Schema for bulk importing DTC codes."""

    codes: List[DTCCreate] = Field(..., min_length=1, max_length=1000)
    overwrite_existing: bool = Field(False, description="Overwrite existing codes with same code")
