# Schemas module
from app.api.v1.schemas.parts import (
    LaborDifficulty,
    PartCategory,
    PartInfo,
    PartsSearchRequest,
    PartsSearchResponse,
    PriceSource,
    RepairCostEstimate,
)
from app.api.v1.schemas.inspection import (
    FailingItem,
    InspectionRequest,
    InspectionResponse,
    InspectionRiskLevel,
    InspectionSeverity,
)
from app.api.v1.schemas.calculator import (
    AlternativeScenario,
    CalculatorRequest,
    CalculatorResponse,
    CostBreakdown,
    ImpactType,
    RecommendationType,
    ValueFactor,
    VehicleCondition,
)
from app.api.v1.schemas.chat import (
    ChatMessage,
    ChatRequest,
    ChatSource,
    ChatStreamEvent,
    VehicleContext,
)
from app.api.v1.schemas.services import (
    Region,
    ServiceSearchParams,
    ServiceSearchResponse,
    ServiceShop,
)

__all__ = [
    # Parts schemas
    "LaborDifficulty",
    "PartCategory",
    "PartInfo",
    "PartsSearchRequest",
    "PartsSearchResponse",
    "PriceSource",
    "RepairCostEstimate",
    # Inspection schemas
    "FailingItem",
    "InspectionRequest",
    "InspectionResponse",
    "InspectionRiskLevel",
    "InspectionSeverity",
    # Calculator schemas
    "AlternativeScenario",
    "CalculatorRequest",
    "CalculatorResponse",
    "CostBreakdown",
    "ImpactType",
    "RecommendationType",
    "ValueFactor",
    "VehicleCondition",
    # Chat schemas
    "ChatMessage",
    "ChatRequest",
    "ChatSource",
    "ChatStreamEvent",
    "VehicleContext",
    # Services schemas
    "Region",
    "ServiceSearchParams",
    "ServiceSearchResponse",
    "ServiceShop",
]
