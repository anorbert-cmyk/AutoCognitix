"""
Műszaki Vizsga (Technical Inspection) Service.

Evaluates vehicle DTC codes against Hungarian MOT inspection categories
and calculates pass/fail risk with cost estimates.

Features:
- DTC prefix to MOT category mapping
- Severity classification (FAIL / WARNING / PASS)
- Risk score calculation weighted by FAIL items
- Cost estimation via PartsPriceService integration
- All 10 Hungarian inspection categories covered

Author: AutoCognitix Team
"""

from typing import Any, Dict, List, Optional, Tuple

from app.api.v1.schemas.inspection import (
    FailingItem,
    InspectionRequest,
    InspectionResponse,
    InspectionRiskLevel,
    InspectionSeverity,
)
from app.core.logging import get_logger
from app.services.parts_price_service import get_parts_price_service

logger = get_logger(__name__)


# =============================================================================
# Hungarian MOT Inspection Categories
# =============================================================================

MOT_CATEGORIES: Dict[str, str] = {
    "emissions": "Kipufogó emisszió",
    "engine_drivetrain": "Motor és hajtáslánc",
    "braking": "Fékrendszer",
    "suspension_steering": "Futómű, kormányzás",
    "safety_systems": "Biztonsági rendszerek",
    "lighting": "Világítás",
    "electrical": "Elektromos rendszer",
    "body_chassis": "Karosszéria és alváz",
    "tires_wheels": "Gumiabroncsok és kerekek",
    "visibility": "Láthatóság (szélvédő, tükrök)",
}


# =============================================================================
# DTC Prefix → Category + Severity Mapping
# =============================================================================

# Each entry: (category_key, default_severity, issue_description_hu)
_DTC_PREFIX_RULES: List[Tuple[str, str, str, InspectionSeverity, str]] = [
    # (prefix_start, prefix_end, category, severity, issue_hu)
    # --- Powertrain (P) codes ---
    # P00xx - P05xx: Emissions-related
    ("P00", "P05", "emissions", InspectionSeverity.FAIL, "Emisszió-szabályozási hiba"),
    # P06xx - P09xx: Engine / drivetrain
    ("P06", "P09", "engine_drivetrain", InspectionSeverity.WARNING, "Motor/hajtáslánc hiba"),
    # --- Chassis (C) codes ---
    # C00xx: Braking / ABS
    ("C00", "C00", "braking", InspectionSeverity.FAIL, "Fékrendszer / ABS hiba"),
    # C01xx - C02xx: Suspension / steering
    (
        "C01",
        "C02",
        "suspension_steering",
        InspectionSeverity.WARNING,
        "Futómű vagy kormányzás hiba",
    ),
    # C03xx+: General chassis
    ("C03", "C0F", "suspension_steering", InspectionSeverity.WARNING, "Alvázrendszer hiba"),
    # --- Body (B) codes ---
    # B00xx: Safety systems (airbag, seatbelt)
    (
        "B00",
        "B00",
        "safety_systems",
        InspectionSeverity.FAIL,
        "Biztonsági rendszer (légzsák/övfeszítő) hiba",
    ),
    # B01xx - B02xx: Lighting
    ("B01", "B02", "lighting", InspectionSeverity.WARNING, "Világítás / jelzőrendszer hiba"),
    # B03xx+: Body electrical
    ("B03", "B0F", "electrical", InspectionSeverity.WARNING, "Karosszéria elektromos hiba"),
    # --- Network (U) codes ---
    # U0xxx - U3xxx: Electrical / communication
    (
        "U00",
        "U3F",
        "electrical",
        InspectionSeverity.WARNING,
        "Kommunikációs / elektromos hálózat hiba",
    ),
]

# Static severity overrides for specific well-known DTC codes
_DTC_SEVERITY_OVERRIDES: Dict[str, Tuple[InspectionSeverity, str]] = {
    # Catalytic converter efficiency - auto FAIL on emissions
    "P0420": (InspectionSeverity.FAIL, "Katalizátor hatásfok alatt (Bank 1)"),
    "P0430": (InspectionSeverity.FAIL, "Katalizátor hatásfok alatt (Bank 2)"),
    # Misfire - FAIL (emissions + safety)
    "P0300": (InspectionSeverity.FAIL, "Több hengeres égéskimaradás"),
    "P0301": (InspectionSeverity.FAIL, "1. henger égéskimaradás"),
    "P0302": (InspectionSeverity.FAIL, "2. henger égéskimaradás"),
    "P0303": (InspectionSeverity.FAIL, "3. henger égéskimaradás"),
    "P0304": (InspectionSeverity.FAIL, "4. henger égéskimaradás"),
    # EGR - FAIL on emissions test
    "P0401": (InspectionSeverity.FAIL, "EGR áramlás elégtelen"),
    # Evaporative system - FAIL
    "P0440": (InspectionSeverity.FAIL, "Párolgásvisszavezető rendszer hiba"),
    "P0442": (InspectionSeverity.WARNING, "Párolgásvisszavezető kis szivárgás"),
    "P0455": (InspectionSeverity.FAIL, "Párolgásvisszavezető nagy szivárgás"),
    # ABS module failure - FAIL
    "C0265": (InspectionSeverity.FAIL, "ABS vezérlőegység belső hiba"),
    # Airbag codes - FAIL
    "B0001": (InspectionSeverity.FAIL, "Légzsák-vezérlő áramköri hiba"),
    "B0002": (InspectionSeverity.FAIL, "Első légzsák hiba"),
    # Headlight - lighting FAIL
    "B1318": (InspectionSeverity.FAIL, "Fényszóró magasságállítás hiba"),
}


# =============================================================================
# Inspection Service
# =============================================================================


class InspectionServiceError(Exception):
    """Custom exception for inspection service errors."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class InspectionService:
    """Service for evaluating vehicle DTC codes against MOT categories.

    Maps each DTC code to a Hungarian inspection category, determines
    severity, estimates repair costs, and computes an overall risk score.
    """

    def __init__(self) -> None:
        """Initialize InspectionService."""
        logger.info("InspectionService inicializálva")

    def _classify_dtc(self, dtc_code: str) -> Tuple[str, InspectionSeverity, str]:
        """Classify a DTC code into MOT category and severity.

        Args:
            dtc_code: Uppercase DTC code (e.g. "P0420").

        Returns:
            Tuple of (category_key, severity, issue_description_hu).
        """
        # Check specific overrides first
        if dtc_code in _DTC_SEVERITY_OVERRIDES:
            severity, issue = _DTC_SEVERITY_OVERRIDES[dtc_code]
            category = self._get_category_for_dtc(dtc_code)
            return category, severity, issue

        # Match against prefix rules
        prefix_3 = dtc_code[:3]
        for start, end, category, severity, issue in _DTC_PREFIX_RULES:
            if start <= prefix_3 <= end:
                return category, severity, f"{issue} ({dtc_code})"

        # Fallback: unknown DTC → electrical warning
        return (
            "electrical",
            InspectionSeverity.WARNING,
            f"Ismeretlen hibakód: {dtc_code}",
        )

    def _get_category_for_dtc(self, dtc_code: str) -> str:
        """Get the MOT category key for a DTC code by prefix.

        Args:
            dtc_code: Uppercase DTC code.

        Returns:
            Category key string.
        """
        prefix_3 = dtc_code[:3]
        for start, end, category, _, _ in _DTC_PREFIX_RULES:
            if start <= prefix_3 <= end:
                return category
        return "electrical"

    def _calculate_risk_score(self, items: List[FailingItem]) -> float:
        """Calculate overall risk score from failing items.

        Weights:
        - FAIL items: 0.3 each (capped contribution at 1.0)
        - WARNING items: 0.1 each
        - Score is clamped to [0.0, 1.0]

        Args:
            items: List of FailingItem results.

        Returns:
            Risk score between 0.0 and 1.0.
        """
        if not items:
            return 0.0

        score = 0.0
        for item in items:
            if item.severity == InspectionSeverity.FAIL:
                score += 0.3
            elif item.severity == InspectionSeverity.WARNING:
                score += 0.1

        return min(score, 1.0)

    def _determine_risk_level(self, risk_score: float, has_fail: bool) -> InspectionRiskLevel:
        """Determine overall risk level from score.

        Args:
            risk_score: Numeric risk score (0-1).
            has_fail: Whether any FAIL-severity items exist.

        Returns:
            InspectionRiskLevel enum value.
        """
        if has_fail or risk_score >= 0.6:
            return InspectionRiskLevel.HIGH
        if risk_score >= 0.3:
            return InspectionRiskLevel.MEDIUM
        return InspectionRiskLevel.LOW

    def _build_recommendations(
        self,
        items: List[FailingItem],
        risk_level: InspectionRiskLevel,
    ) -> List[str]:
        """Build Hungarian-language recommendations.

        Args:
            items: List of failing/warning items.
            risk_level: Overall risk level.

        Returns:
            List of recommendation strings in Hungarian.
        """
        recommendations: List[str] = []

        fail_count = sum(1 for i in items if i.severity == InspectionSeverity.FAIL)
        warn_count = sum(1 for i in items if i.severity == InspectionSeverity.WARNING)

        if fail_count > 0:
            recommendations.append(
                f"{fail_count} kritikus hiba - a műszaki vizsga "
                f"NEM lenne sikeres a jelenlegi állapotban."
            )
        if warn_count > 0:
            recommendations.append(
                f"{warn_count} figyelmeztető tétel - javítás javasolt a vizsga előtt."
            )

        # Category-specific advice
        categories_seen = {i.category for i in items}
        if "emissions" in categories_seen:
            recommendations.append(
                "Emisszió-szabályozás javítása szükséges a szonda/katalizátor/EGR rendszerben."
            )
        if "braking" in categories_seen:
            recommendations.append(
                "Fékrendszer vizsgálata és javítása SÜRGŐS - közlekedésbiztonsági kockázat!"
            )
        if "safety_systems" in categories_seen:
            recommendations.append(
                "Biztonsági rendszer (légzsák/övfeszítő) javítása kötelező a vizsga előtt."
            )

        if risk_level == InspectionRiskLevel.LOW:
            recommendations.append(
                "Alacsony kockázat - a jármű valószínűleg átmenne a műszaki vizsgán."
            )

        return recommendations

    async def evaluate(self, request: InspectionRequest) -> InspectionResponse:
        """Evaluate DTC codes against MOT inspection categories.

        Args:
            request: InspectionRequest with vehicle info and DTC codes.

        Returns:
            InspectionResponse with risk assessment and failing items.

        Raises:
            InspectionServiceError: If evaluation fails unexpectedly.
        """
        logger.info(
            "Műszaki vizsga értékelés indítása",
            extra={
                "vehicle": (
                    f"{request.vehicle_make} {request.vehicle_model} {request.vehicle_year}"
                ),
                "dtc_count": len(request.dtc_codes),
            },
        )

        failing_items: List[FailingItem] = []
        affected_categories: set = set()

        # Classify each DTC code
        for dtc_code in request.dtc_codes:
            category, severity, issue = self._classify_dtc(dtc_code)
            affected_categories.add(category)

            # Get cost estimates from PartsPriceService
            cost_min = 0
            cost_max = 0
            fix_recommendation = f"{issue} - szakszervizes vizsgálat szükséges"

            try:
                price_service = get_parts_price_service()
                parts = await price_service.get_parts_for_dtc(
                    dtc_code=dtc_code,
                    vehicle_make=request.vehicle_make,
                    vehicle_model=request.vehicle_model,
                    vehicle_year=request.vehicle_year,
                )
                if parts:
                    cost_min = sum(p.get("price_min", 0) for p in parts)
                    cost_max = sum(p.get("price_max", 0) for p in parts)
                    part_names = [p.get("name", "?") for p in parts[:3]]
                    fix_recommendation = f"{issue} - javasolt alkatrészek: {', '.join(part_names)}"
            except Exception as exc:
                logger.warning(
                    f"Alkatrész ár lekérés sikertelen: {dtc_code} - {exc}",
                    extra={"dtc_code": dtc_code},
                )

            failing_items.append(
                FailingItem(
                    category=category,
                    category_hu=MOT_CATEGORIES.get(category, category),
                    issue=issue,
                    related_dtc=dtc_code,
                    severity=severity,
                    fix_recommendation=fix_recommendation,
                    estimated_cost_min=cost_min,
                    estimated_cost_max=cost_max,
                )
            )

        # Determine passing categories
        all_categories = set(MOT_CATEGORIES.keys())
        passing_categories = sorted(all_categories - affected_categories)

        # Calculate risk
        risk_score = self._calculate_risk_score(failing_items)
        has_fail = any(i.severity == InspectionSeverity.FAIL for i in failing_items)
        risk_level = self._determine_risk_level(risk_score, has_fail)

        # Build recommendations
        recommendations = self._build_recommendations(failing_items, risk_level)

        # Total costs
        total_min = sum(i.estimated_cost_min for i in failing_items)
        total_max = sum(i.estimated_cost_max for i in failing_items)

        # Vehicle info string
        engine_str = f" {request.vehicle_engine}" if request.vehicle_engine else ""
        mileage_str = (
            f", {request.mileage_km:,} km".replace(",", " ")
            if request.mileage_km is not None
            else ""
        )
        vehicle_info = (
            f"{request.vehicle_make} {request.vehicle_model} "
            f"{request.vehicle_year}{engine_str}{mileage_str}"
        )

        logger.info(
            "Műszaki vizsga értékelés kész",
            extra={
                "risk_level": risk_level.value,
                "risk_score": risk_score,
                "fail_count": sum(
                    1 for i in failing_items if i.severity == InspectionSeverity.FAIL
                ),
                "warning_count": sum(
                    1 for i in failing_items if i.severity == InspectionSeverity.WARNING
                ),
            },
        )

        return InspectionResponse(
            overall_risk=risk_level,
            risk_score=round(risk_score, 2),
            failing_items=failing_items,
            passing_categories=passing_categories,
            recommendations=recommendations,
            estimated_total_fix_cost_min=total_min,
            estimated_total_fix_cost_max=total_max,
            vehicle_info=vehicle_info,
            dtc_count=len(request.dtc_codes),
        )


# =============================================================================
# Singleton Factory
# =============================================================================

_service_instance: Optional[InspectionService] = None


def get_inspection_service() -> InspectionService:
    """Get or create the singleton InspectionService instance.

    Returns:
        InspectionService singleton instance.
    """
    global _service_instance
    if _service_instance is None:
        _service_instance = InspectionService()
    return _service_instance
