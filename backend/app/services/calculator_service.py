"""
Calculator Service - "Megeri megjavitani?" (Worth Repairing?) calculator.

Jarmu ertekbecsles es javitasi koltseg osszehasonlitas
a magyar autopiaci adatok alapjan.

Features:
- Magyar piaci jarmu ertekbecsles amortizacios modell alapjan
- Premium marka kulon amortizacio
- Kilometerallas korrekciok
- Allapot szorzok
- Javitasi erdemesseg kiertekeles
- Alternativ forgatokonyvek generalasa

Author: AutoCognitix Team
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from app.core.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Static Data - Hungarian Vehicle Market
# =============================================================================

# MSRP ranges by vehicle class (HUF, new vehicle prices)
VEHICLE_CLASS_MSRP: Dict[str, Dict[str, int]] = {
    "subcompact": {"min": 4_000_000, "max": 6_500_000},
    "compact": {"min": 5_500_000, "max": 8_500_000},
    "midsize": {"min": 7_500_000, "max": 12_000_000},
    "suv_compact": {"min": 5_500_000, "max": 8_500_000},
    "suv_midsize": {"min": 11_000_000, "max": 18_000_000},
    "premium": {"min": 12_000_000, "max": 25_000_000},
}

# Make-to-class mapping
MAKE_CLASS_MAPPING: Dict[str, str] = {
    "DACIA": "subcompact",
    "SUZUKI": "subcompact",
    "KIA": "compact",
    "HYUNDAI": "compact",
    "VOLKSWAGEN": "compact",
    "VW": "compact",
    "TOYOTA": "compact",
    "FORD": "compact",
    "OPEL": "compact",
    "SKODA": "compact",
    "RENAULT": "compact",
    "PEUGEOT": "compact",
    "CITROEN": "compact",
    "SEAT": "compact",
    "FIAT": "subcompact",
    "MAZDA": "compact",
    "HONDA": "compact",
    "NISSAN": "compact",
    "MITSUBISHI": "suv_compact",
    "VOLVO": "midsize",
    "BMW": "premium",
    "AUDI": "premium",
    "MERCEDES": "premium",
    "MERCEDES-BENZ": "premium",
    "PORSCHE": "premium",
    "LEXUS": "premium",
    "JAGUAR": "premium",
    "LAND ROVER": "suv_midsize",
}

# Standard depreciation curve (fraction of MSRP retained)
STANDARD_DEPRECIATION: Dict[int, float] = {
    1: 0.85,
    2: 0.75,
    3: 0.68,
    4: 0.62,
    5: 0.57,
    6: 0.52,
    7: 0.48,
    8: 0.44,
    9: 0.41,
    10: 0.38,
}

# Premium depreciation curve (BMW, Audi, Mercedes)
PREMIUM_DEPRECIATION: Dict[int, float] = {
    1: 0.80,
    2: 0.68,
}

PREMIUM_MAKES = {"BMW", "AUDI", "MERCEDES", "MERCEDES-BENZ", "PORSCHE"}

# Condition multipliers
CONDITION_MULTIPLIERS: Dict[str, float] = {
    "excellent": 1.15,
    "good": 1.00,
    "fair": 0.88,
    "poor": 0.70,
}

# Hungarian average annual mileage (km)
HUNGARIAN_AVG_ANNUAL_KM = 15_000

# Fuel type value adjustments
FUEL_TYPE_ADJUSTMENTS: Dict[str, float] = {
    "benzin": 1.00,
    "dizel": 0.95,
    "elektromos": 1.10,
    "hybrid": 1.08,
    "lpg": 0.90,
}


# =============================================================================
# Calculator Service
# =============================================================================


class CalculatorService:
    """
    Jarmu ertekbecsles es javitasi erdemesseg szamitas.

    A szolgaltatas a magyar hasznaltauto piac statisztikai
    adatain alapul (amortizacio, kilometerallas, allapot).
    """

    _instance: Optional["CalculatorService"] = None
    _initialized: bool = False

    def __new__(cls) -> "CalculatorService":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize service."""
        if self._initialized:
            return
        self._initialized = True
        logger.info("CalculatorService inicializalva")

    def _get_vehicle_class(self, make: str) -> str:
        """Determine vehicle class from make."""
        return MAKE_CLASS_MAPPING.get(make.upper(), "compact")

    def _is_premium(self, make: str) -> bool:
        """Check if make is a premium brand."""
        return make.upper() in PREMIUM_MAKES

    def _get_depreciation_factor(self, vehicle_age: int, make: str) -> float:
        """
        Calculate depreciation factor based on age and brand.

        Premium brands depreciate faster initially.
        """
        if vehicle_age <= 0:
            return 1.0

        if self._is_premium(make):
            # Premium depreciation: Year 1=0.80, Year 2=0.68,
            # then -0.06/year, floor at 0.08
            if vehicle_age in PREMIUM_DEPRECIATION:
                return PREMIUM_DEPRECIATION[vehicle_age]
            if vehicle_age > 2:
                factor = 0.68 - (0.06 * (vehicle_age - 2))
                return max(factor, 0.08)
        else:
            # Standard depreciation from lookup table
            if vehicle_age in STANDARD_DEPRECIATION:
                return STANDARD_DEPRECIATION[vehicle_age]
            # Years 11-20: subtract 0.02 per year, floor at 0.10
            if vehicle_age > 10:
                factor = 0.38 - (0.02 * (vehicle_age - 10))
                return max(factor, 0.10)

        return 0.10

    def _get_mileage_adjustment(self, mileage_km: int, vehicle_age: int) -> float:
        """
        Calculate mileage-based value adjustment.

        Hungarian average is 15,000 km/year.
        Above average: -1% per 5,000 km excess (capped at -30%).
        Below average: no bonus (conservative estimate).
        """
        if vehicle_age <= 0:
            return 0.0

        expected_km = HUNGARIAN_AVG_ANNUAL_KM * vehicle_age
        excess_km = mileage_km - expected_km

        if excess_km <= 0:
            return 0.0

        # -1% per 5,000 km excess, capped at -30%
        penalty_pct = (excess_km / 5000) * 0.01
        return max(-0.30, -penalty_pct)

    def estimate_vehicle_value(
        self,
        make: str,
        model: str,
        year: int,
        mileage_km: int,
        condition: str,
    ) -> Dict[str, int]:
        """
        Estimate current vehicle market value.

        Args:
            make: Vehicle manufacturer
            model: Vehicle model
            year: Manufacturing year
            mileage_km: Current odometer reading (km)
            condition: Vehicle condition (excellent/good/fair/poor)

        Returns:
            Dict with min, max, avg vehicle value in HUF
        """
        current_year = datetime.now(timezone.utc).year
        vehicle_age = current_year - year

        # Get MSRP range for vehicle class
        vehicle_class = self._get_vehicle_class(make)
        msrp = VEHICLE_CLASS_MSRP.get(vehicle_class, VEHICLE_CLASS_MSRP["compact"])

        # Apply depreciation
        depreciation = self._get_depreciation_factor(vehicle_age, make)

        # Apply condition multiplier
        cond_mult = CONDITION_MULTIPLIERS.get(condition, 1.0)

        # Apply mileage adjustment
        mileage_adj = self._get_mileage_adjustment(mileage_km, vehicle_age)

        # Calculate value range
        combined_factor = depreciation * cond_mult * (1.0 + mileage_adj)
        combined_factor = max(combined_factor, 0.05)

        value_min = int(msrp["min"] * combined_factor)
        value_max = int(msrp["max"] * combined_factor)
        value_avg = (value_min + value_max) // 2

        # Ensure minimum floor (scrap value)
        value_min = max(value_min, 100_000)
        value_max = max(value_max, value_min)
        value_avg = max(value_avg, value_min)

        logger.info(
            "Jarmu ertekbecsles: %s %s %d, %d km, %s allapot -> %s - %s HUF (atlag: %s)",
            make,
            model,
            year,
            mileage_km,
            condition,
            f"{value_min:,}",
            f"{value_max:,}",
            f"{value_avg:,}",
        )

        return {
            "min": value_min,
            "max": value_max,
            "avg": value_avg,
        }

    def evaluate_repair_worthiness(
        self, vehicle_value_avg: int, repair_cost: int
    ) -> Dict[str, Union[str, float]]:
        """
        Evaluate whether repair is worth the cost.

        Decision thresholds:
        - ratio < 0.35 -> "repair" (Javitas ajanlott)
        - 0.35 <= ratio <= 0.65 -> "sell" (Eladas fontolora veendo)
        - ratio > 0.65 -> "scrap" (Roncskent torteno ertekesites)

        Args:
            vehicle_value_avg: Average estimated vehicle value (HUF)
            repair_cost: Estimated repair cost (HUF)

        Returns:
            Dict with recommendation and recommendation_text
        """
        if vehicle_value_avg <= 0:
            return {
                "recommendation": "scrap",
                "recommendation_text": (
                    "A jarmu becsult erteke tul alacsony. Roncskent torteno ertekesites javasolt."
                ),
                "ratio": 1.0,
            }

        ratio = repair_cost / vehicle_value_avg

        if ratio < 0.35:
            recommendation = "repair"
            text = (
                "Javitas ajanlott. A javitasi koltseg "
                f"mindossze {ratio:.0%}-a a jarmu ertekenek, "
                "igy a javitas gazdasagos."
            )
        elif ratio <= 0.65:
            recommendation = "sell"
            text = (
                "Eladas fontolora veendo. A javitasi koltseg "
                f"({ratio:.0%} a jarmu ertekenek) jelentos, "
                "erdemes merlegel a javitas es az eladas kozott."
            )
        else:
            recommendation = "scrap"
            text = (
                "Roncskent torteno ertekesites javasolt. "
                f"A javitasi koltseg ({ratio:.0%} a jarmu "
                "ertekenek) meghaladja a gazdasagossagi hatart."
            )

        return {
            "recommendation": recommendation,
            "recommendation_text": text,
            "ratio": round(ratio, 4),
        }

    def generate_factors(
        self,
        make: str,
        model: str,
        year: int,
        mileage_km: int,
        condition: str,
        fuel_type: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Generate value factors affecting the assessment.

        Args:
            make: Vehicle manufacturer
            model: Vehicle model
            year: Manufacturing year
            mileage_km: Odometer reading (km)
            condition: Vehicle condition
            fuel_type: Fuel type (optional)

        Returns:
            List of factor dicts with name, impact, description
        """
        current_year = datetime.now(timezone.utc).year
        vehicle_age = current_year - year
        factors: List[Dict[str, str]] = []

        # Vehicle age factor
        if vehicle_age <= 3:
            factors.append(
                {
                    "name": "Jarmu kora",
                    "impact": "positive",
                    "description": (
                        f"A jarmu mindossze {vehicle_age} eves, "
                        "ami kedvezo az ertekmegorzes szempontjabol."
                    ),
                }
            )
        elif vehicle_age <= 7:
            factors.append(
                {
                    "name": "Jarmu kora",
                    "impact": "negative",
                    "description": (
                        f"A {vehicle_age} eves jarmu az atlagos amortizacios savban van."
                    ),
                }
            )
        else:
            factors.append(
                {
                    "name": "Jarmu kora",
                    "impact": "negative",
                    "description": (f"A {vehicle_age} eves jarmu jelentos ertekcsokkenest mutat."),
                }
            )

        # Mileage factor
        expected_km = HUNGARIAN_AVG_ANNUAL_KM * max(vehicle_age, 1)
        if mileage_km > expected_km * 1.2:
            factors.append(
                {
                    "name": "Kilometerallas",
                    "impact": "negative",
                    "description": (
                        f"A {mileage_km:,} km-es futasteljesitmeny "
                        f"meghaladja az elvarhato {expected_km:,} km-t."
                    ),
                }
            )
        elif mileage_km < expected_km * 0.8:
            factors.append(
                {
                    "name": "Kilometerallas",
                    "impact": "positive",
                    "description": (
                        f"A {mileage_km:,} km-es futasteljesitmeny "
                        f"az elvarhato {expected_km:,} km alatt van."
                    ),
                }
            )

        # Condition factor
        condition_labels = {
            "excellent": "kimagaslo",
            "good": "jo",
            "fair": "elfogadhato",
            "poor": "gyenge",
        }
        cond_label = condition_labels.get(condition, condition)
        cond_impact = "positive" if condition in ("excellent", "good") else "negative"
        factors.append(
            {
                "name": "Altalanos allapot",
                "impact": cond_impact,
                "description": (
                    f"A jarmu allapota {cond_label}, ami "
                    + (
                        "kedvezoen hat az ertekere."
                        if cond_impact == "positive"
                        else "csokkenti az erteket."
                    )
                ),
            }
        )

        # Brand premium factor
        if self._is_premium(make):
            factors.append(
                {
                    "name": "Marka premium",
                    "impact": "positive",
                    "description": (
                        f"A {make} premium marka, ami magasabb "
                        "MSRP-vel rendelkezik, de gyorsabb "
                        "kezdeti ertekcsokkenesre is szamithatsz."
                    ),
                }
            )

        # Fuel type factor
        if fuel_type:
            fuel_adj = FUEL_TYPE_ADJUSTMENTS.get(fuel_type.lower(), 1.0)
            if fuel_adj > 1.0:
                factors.append(
                    {
                        "name": "Uzemanyag tipus",
                        "impact": "positive",
                        "description": (
                            f"A {fuel_type} uzemanyag tipusu jarmuvek iranti kereslet novekedik."
                        ),
                    }
                )
            elif fuel_adj < 1.0:
                factors.append(
                    {
                        "name": "Uzemanyag tipus",
                        "impact": "negative",
                        "description": (
                            f"A {fuel_type} uzemanyag tipusu jarmuvek iranti kereslet csokkeno."
                        ),
                    }
                )

        return factors

    def generate_alternative_scenarios(
        self,
        vehicle_value_avg: int,
        repair_cost: int,
        condition: str,
    ) -> List[Dict[str, Any]]:
        """
        Generate alternative scenarios for the vehicle owner.

        Args:
            vehicle_value_avg: Average vehicle value (HUF)
            repair_cost: Estimated repair cost (HUF)
            condition: Current vehicle condition

        Returns:
            List of scenario dicts
        """
        scenarios: List[Dict[str, Any]] = []

        # Scenario 1: Sell as-is
        condition_discount = {
            "excellent": 0.95,
            "good": 0.90,
            "fair": 0.75,
            "poor": 0.55,
        }
        discount = condition_discount.get(condition, 0.75)
        sell_as_is_value = int(vehicle_value_avg * discount)
        scenarios.append(
            {
                "scenario": "Eladas jelenlegi allapotban",
                "description": (
                    "A jarmu jelenlegi allapotaban torteno "
                    "ertekesitese a hasznaltauto piacon. "
                    "Az ar a hiba mertekenek megfeleloen "
                    "csokkentett."
                ),
                "estimated_value": max(sell_as_is_value, 100_000),
            }
        )

        # Scenario 2: Repair then sell
        repaired_value = int(vehicle_value_avg * 1.0)
        net_after_repair = repaired_value - repair_cost
        scenarios.append(
            {
                "scenario": "Javitas utani eladas",
                "description": (
                    "A jarmu javitasat kovetoen torteno "
                    "ertekesitese. A javitasi koltseg "
                    f"({repair_cost:,} HUF) levonasra kerul "
                    "a teljes ertekbol."
                ),
                "estimated_value": max(net_after_repair, 0),
            }
        )

        # Scenario 3: Scrap
        scrap_value = max(int(vehicle_value_avg * 0.08), 50_000)
        scenarios.append(
            {
                "scenario": "Roncskent ertekesites",
                "description": (
                    "A jarmu roncs/bontott jarmukent torteno "
                    "ertekesitese. Az alkatreszek es femhulladek "
                    "erteket tartalmazza."
                ),
                "estimated_value": scrap_value,
            }
        )

        return scenarios

    async def calculate(
        self,
        vehicle_make: str,
        vehicle_model: str,
        vehicle_year: int,
        mileage_km: int,
        condition: str,
        repair_cost_huf: Optional[int] = None,
        diagnosis_id: Optional[str] = None,
        fuel_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Main calculator entry point.

        Performs vehicle valuation and repair worthiness analysis.

        Args:
            vehicle_make: Vehicle manufacturer
            vehicle_model: Vehicle model
            vehicle_year: Manufacturing year
            mileage_km: Odometer reading (km)
            condition: Vehicle condition
            repair_cost_huf: Known repair cost (optional)
            diagnosis_id: Previous diagnosis ID (optional)
            fuel_type: Fuel type (optional)

        Returns:
            Complete calculator response dict
        """
        logger.info(
            "Kalkulator inditas: %s %s %d, %d km, %s",
            vehicle_make,
            vehicle_model,
            vehicle_year,
            mileage_km,
            condition,
        )

        # Step 1: Estimate vehicle value
        value = self.estimate_vehicle_value(
            make=vehicle_make,
            model=vehicle_model,
            year=vehicle_year,
            mileage_km=mileage_km,
            condition=condition,
        )

        # Apply fuel type adjustment
        if fuel_type:
            fuel_adj = FUEL_TYPE_ADJUSTMENTS.get(fuel_type.lower(), 1.0)
            value["min"] = int(value["min"] * fuel_adj)
            value["max"] = int(value["max"] * fuel_adj)
            value["avg"] = int(value["avg"] * fuel_adj)

        # Step 2: Determine repair cost
        repair_cost_min = 0
        repair_cost_max = 0
        parts_cost = 0
        labor_cost = 0
        confidence = 0.60

        if repair_cost_huf is not None:
            # User provided explicit repair cost
            repair_cost_min = int(repair_cost_huf * 0.85)
            repair_cost_max = int(repair_cost_huf * 1.15)
            parts_cost = int(repair_cost_huf * 0.60)
            labor_cost = int(repair_cost_huf * 0.40)
            confidence = 0.80
        elif diagnosis_id:
            # Try to fetch from diagnosis session
            cost_data = await self._fetch_diagnosis_cost(diagnosis_id)
            if cost_data:
                repair_cost_min = cost_data.get("total_min", 0)
                repair_cost_max = cost_data.get("total_max", 0)
                parts_cost = cost_data.get("parts_avg", 0)
                labor_cost = cost_data.get("labor_avg", 0)
                confidence = 0.75
            else:
                logger.warning(
                    "Diagnosztika nem talalhato: %s, fallback koltseg becsles",
                    diagnosis_id,
                )
                repair_cost_min = 50_000
                repair_cost_max = 300_000
                confidence = 0.40
        else:
            # No cost info — use generic fallback
            repair_cost_min = 50_000
            repair_cost_max = 300_000
            confidence = 0.40

        repair_cost_avg = (repair_cost_min + repair_cost_max) // 2

        # Step 3: Evaluate repair worthiness
        evaluation = self.evaluate_repair_worthiness(
            vehicle_value_avg=value["avg"],
            repair_cost=repair_cost_avg,
        )

        # Step 4: Generate factors
        factors = self.generate_factors(
            make=vehicle_make,
            model=vehicle_model,
            year=vehicle_year,
            mileage_km=mileage_km,
            condition=condition,
            fuel_type=fuel_type,
        )

        # Step 5: Generate alternative scenarios
        scenarios = self.generate_alternative_scenarios(
            vehicle_value_avg=value["avg"],
            repair_cost=repair_cost_avg,
            condition=condition,
        )

        result = {
            "vehicle_value_min": value["min"],
            "vehicle_value_max": value["max"],
            "vehicle_value_avg": value["avg"],
            "repair_cost_min": repair_cost_min,
            "repair_cost_max": repair_cost_max,
            "ratio": evaluation["ratio"],
            "recommendation": evaluation["recommendation"],
            "recommendation_text": evaluation["recommendation_text"],
            "breakdown": {
                "parts_cost": parts_cost,
                "labor_cost": labor_cost,
                "additional_costs": 0,
            },
            "factors": factors,
            "alternative_scenarios": scenarios,
            "confidence_score": confidence,
            "currency": "HUF",
        }

        logger.info(
            "Kalkulator eredmeny: %s %s %d -> %s (ratio: %.2f)",
            vehicle_make,
            vehicle_model,
            vehicle_year,
            evaluation["recommendation"],
            evaluation["ratio"],
        )

        return result

    async def _fetch_diagnosis_cost(self, diagnosis_id: str) -> Optional[Dict[str, int]]:
        """
        Fetch repair cost estimate from a previous diagnosis session.

        Args:
            diagnosis_id: UUID of the diagnosis session

        Returns:
            Dict with cost breakdown or None if not found
        """
        try:
            from app.db.postgres.session import async_session_maker

            async with async_session_maker() as session:
                from sqlalchemy import select

                from app.db.postgres.models import DiagnosisSession

                stmt = select(DiagnosisSession).where(
                    DiagnosisSession.id == diagnosis_id,
                    DiagnosisSession.is_deleted.is_(False),
                )
                result = await session.execute(stmt)
                diag = result.scalar_one_or_none()

                if not diag:
                    return None

                # Extract cost from diagnosis_result JSONB
                diag_result = diag.diagnosis_result or {}
                cost_estimate = diag_result.get("total_cost_estimate", {})

                if not cost_estimate:
                    return None

                parts_min = cost_estimate.get("parts_min", 0)
                parts_max = cost_estimate.get("parts_max", 0)
                labor_min = cost_estimate.get("labor_min", 0)
                labor_max = cost_estimate.get("labor_max", 0)

                return {
                    "total_min": cost_estimate.get("total_min", parts_min + labor_min),
                    "total_max": cost_estimate.get("total_max", parts_max + labor_max),
                    "parts_avg": (parts_min + parts_max) // 2,
                    "labor_avg": (labor_min + labor_max) // 2,
                }

        except Exception as e:
            logger.error("Diagnosztika koltseg lekeres hiba: %s", str(e))
            return None


# =============================================================================
# Module-level Functions
# =============================================================================

_service_instance: Optional[CalculatorService] = None


def get_calculator_service() -> CalculatorService:
    """
    Get singleton service instance.

    Returns:
        CalculatorService instance
    """
    global _service_instance
    if _service_instance is None:
        _service_instance = CalculatorService()
    return _service_instance
