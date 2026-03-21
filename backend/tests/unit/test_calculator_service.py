"""
Unit tests for CalculatorService.

Tests cover:
- Vehicle value estimation
- Depreciation calculations (standard and premium)
- Mileage adjustments
- Condition multipliers
- Repair worthiness evaluation
- Factor generation
- Alternative scenario generation
- Main calculate() async method
- Singleton pattern
- Edge cases (zero/negative values, unknown makes, missing data)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.calculator_service import (
    CONDITION_MULTIPLIERS,
    FUEL_TYPE_ADJUSTMENTS,
    HUNGARIAN_AVG_ANNUAL_KM,
    MAKE_CLASS_MAPPING,
    PREMIUM_MAKES,
    STANDARD_DEPRECIATION,
    VEHICLE_CLASS_MSRP,
    CalculatorService,
    get_calculator_service,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset singleton between tests."""
    CalculatorService._instance = None
    CalculatorService._initialized = False
    yield
    CalculatorService._instance = None
    CalculatorService._initialized = False


@pytest.fixture
def svc() -> CalculatorService:
    return CalculatorService()


# ---------------------------------------------------------------------------
# Singleton / get_calculator_service
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_singleton_returns_same_instance(self):
        a = CalculatorService()
        b = CalculatorService()
        assert a is b

    def test_get_calculator_service(self, _reset_singleton):
        import app.services.calculator_service as mod

        mod._service_instance = None
        svc = get_calculator_service()
        assert isinstance(svc, CalculatorService)
        assert get_calculator_service() is svc


# ---------------------------------------------------------------------------
# _get_vehicle_class
# ---------------------------------------------------------------------------


class TestGetVehicleClass:
    def test_known_make(self, svc):
        assert svc._get_vehicle_class("BMW") == "premium"

    def test_case_insensitive(self, svc):
        assert svc._get_vehicle_class("bmw") == "premium"

    def test_unknown_defaults_to_compact(self, svc):
        assert svc._get_vehicle_class("UNKNOWN_BRAND") == "compact"


# ---------------------------------------------------------------------------
# _is_premium
# ---------------------------------------------------------------------------


class TestIsPremium:
    def test_premium_brands(self, svc):
        for make in PREMIUM_MAKES:
            assert svc._is_premium(make) is True

    def test_non_premium(self, svc):
        assert svc._is_premium("DACIA") is False


# ---------------------------------------------------------------------------
# _get_depreciation_factor
# ---------------------------------------------------------------------------


class TestDepreciation:
    def test_new_vehicle_no_depreciation(self, svc):
        assert svc._get_depreciation_factor(0, "TOYOTA") == 1.0

    def test_negative_age(self, svc):
        assert svc._get_depreciation_factor(-1, "TOYOTA") == 1.0

    def test_standard_lookup(self, svc):
        for age, expected in STANDARD_DEPRECIATION.items():
            assert svc._get_depreciation_factor(age, "TOYOTA") == expected

    def test_standard_beyond_table(self, svc):
        # Year 12 -> 0.38 - 0.02*2 = 0.34
        assert svc._get_depreciation_factor(12, "TOYOTA") == pytest.approx(0.34)

    def test_standard_floor(self, svc):
        # Very old car should not go below 0.10
        assert svc._get_depreciation_factor(30, "TOYOTA") == 0.10

    def test_premium_year1(self, svc):
        assert svc._get_depreciation_factor(1, "BMW") == 0.80

    def test_premium_year2(self, svc):
        assert svc._get_depreciation_factor(2, "AUDI") == 0.68

    def test_premium_beyond_table(self, svc):
        # Year 4 -> 0.68 - 0.06*2 = 0.56
        assert svc._get_depreciation_factor(4, "BMW") == pytest.approx(0.56)

    def test_premium_floor(self, svc):
        assert svc._get_depreciation_factor(50, "BMW") == 0.08


# ---------------------------------------------------------------------------
# _get_mileage_adjustment
# ---------------------------------------------------------------------------


class TestMileageAdjustment:
    def test_zero_age(self, svc):
        assert svc._get_mileage_adjustment(100_000, 0) == 0.0

    def test_below_average(self, svc):
        # Below average -> no adjustment
        assert svc._get_mileage_adjustment(10_000, 5) == 0.0

    def test_above_average(self, svc):
        # 5 years, expected = 75000, actual = 100000 -> excess 25000
        # penalty = (25000/5000)*0.01 = 0.05
        adj = svc._get_mileage_adjustment(100_000, 5)
        assert adj == pytest.approx(-0.05)

    def test_cap_at_minus_30(self, svc):
        adj = svc._get_mileage_adjustment(500_000, 1)
        assert adj == -0.30


# ---------------------------------------------------------------------------
# estimate_vehicle_value
# ---------------------------------------------------------------------------


class TestEstimateVehicleValue:
    def test_returns_min_max_avg(self, svc):
        result = svc.estimate_vehicle_value("TOYOTA", "Corolla", 2020, 60_000, "good")
        assert "min" in result
        assert "max" in result
        assert "avg" in result
        assert result["min"] <= result["avg"] <= result["max"]

    def test_minimum_floor(self, svc):
        # Very old, high mileage, poor condition
        result = svc.estimate_vehicle_value("DACIA", "Logan", 1990, 900_000, "poor")
        assert result["min"] >= 100_000

    def test_premium_vs_standard(self, svc):
        standard = svc.estimate_vehicle_value("TOYOTA", "Corolla", 2020, 50_000, "good")
        premium = svc.estimate_vehicle_value("BMW", "3 Series", 2020, 50_000, "good")
        # Premium class has higher MSRP range
        assert premium["avg"] > standard["avg"]

    def test_excellent_vs_poor_condition(self, svc):
        excellent = svc.estimate_vehicle_value("VW", "Golf", 2020, 50_000, "excellent")
        poor = svc.estimate_vehicle_value("VW", "Golf", 2020, 50_000, "poor")
        assert excellent["avg"] > poor["avg"]


# ---------------------------------------------------------------------------
# evaluate_repair_worthiness
# ---------------------------------------------------------------------------


class TestRepairWorthiness:
    def test_repair_recommended(self, svc):
        result = svc.evaluate_repair_worthiness(3_000_000, 500_000)
        assert result["recommendation"] == "repair"
        assert result["ratio"] < 0.35

    def test_sell_recommended(self, svc):
        result = svc.evaluate_repair_worthiness(1_000_000, 500_000)
        assert result["recommendation"] == "sell"

    def test_scrap_recommended(self, svc):
        result = svc.evaluate_repair_worthiness(500_000, 500_000)
        assert result["recommendation"] == "scrap"
        assert result["ratio"] >= 0.65

    def test_zero_vehicle_value(self, svc):
        result = svc.evaluate_repair_worthiness(0, 100_000)
        assert result["recommendation"] == "scrap"
        assert result["ratio"] == 1.0

    def test_negative_vehicle_value(self, svc):
        result = svc.evaluate_repair_worthiness(-100, 100_000)
        assert result["recommendation"] == "scrap"

    def test_ratio_rounded(self, svc):
        result = svc.evaluate_repair_worthiness(3_000_000, 1_000_000)
        # ratio = 1000000/3000000 = 0.3333...
        assert result["ratio"] == round(1_000_000 / 3_000_000, 4)


# ---------------------------------------------------------------------------
# generate_factors
# ---------------------------------------------------------------------------


class TestGenerateFactors:
    def test_returns_list(self, svc):
        factors = svc.generate_factors("VW", "Golf", 2024, 20_000, "good")
        assert isinstance(factors, list)
        assert len(factors) >= 1

    def test_young_vehicle_positive(self, svc):
        factors = svc.generate_factors("VW", "Golf", 2024, 10_000, "good")
        age_factor = next(f for f in factors if f["name"] == "Jarmu kora")
        assert age_factor["impact"] == "positive"

    def test_old_vehicle_negative(self, svc):
        factors = svc.generate_factors("VW", "Golf", 2010, 200_000, "fair")
        age_factor = next(f for f in factors if f["name"] == "Jarmu kora")
        assert age_factor["impact"] == "negative"

    def test_premium_brand_factor(self, svc):
        factors = svc.generate_factors("BMW", "3 Series", 2020, 80_000, "good")
        premium_factors = [f for f in factors if f["name"] == "Marka premium"]
        assert len(premium_factors) == 1

    def test_no_premium_for_standard(self, svc):
        factors = svc.generate_factors("DACIA", "Logan", 2020, 50_000, "good")
        premium_factors = [f for f in factors if f["name"] == "Marka premium"]
        assert len(premium_factors) == 0

    def test_high_mileage_negative(self, svc):
        factors = svc.generate_factors("VW", "Golf", 2022, 200_000, "good")
        km_factors = [f for f in factors if f["name"] == "Kilometerallas"]
        assert len(km_factors) == 1
        assert km_factors[0]["impact"] == "negative"

    def test_low_mileage_positive(self, svc):
        factors = svc.generate_factors("VW", "Golf", 2020, 10_000, "good")
        km_factors = [f for f in factors if f["name"] == "Kilometerallas"]
        assert len(km_factors) == 1
        assert km_factors[0]["impact"] == "positive"

    def test_fuel_type_elektromos_positive(self, svc):
        factors = svc.generate_factors("VW", "Golf", 2022, 30_000, "good", fuel_type="elektromos")
        fuel_factors = [f for f in factors if f["name"] == "Uzemanyag tipus"]
        assert len(fuel_factors) == 1
        assert fuel_factors[0]["impact"] == "positive"

    def test_fuel_type_lpg_negative(self, svc):
        factors = svc.generate_factors("VW", "Golf", 2022, 30_000, "good", fuel_type="lpg")
        fuel_factors = [f for f in factors if f["name"] == "Uzemanyag tipus"]
        assert len(fuel_factors) == 1
        assert fuel_factors[0]["impact"] == "negative"

    def test_fuel_type_benzin_neutral(self, svc):
        factors = svc.generate_factors("VW", "Golf", 2022, 30_000, "good", fuel_type="benzin")
        fuel_factors = [f for f in factors if f["name"] == "Uzemanyag tipus"]
        assert len(fuel_factors) == 0  # 1.0 adjustment -> no factor


# ---------------------------------------------------------------------------
# generate_alternative_scenarios
# ---------------------------------------------------------------------------


class TestAlternativeScenarios:
    def test_returns_three_scenarios(self, svc):
        scenarios = svc.generate_alternative_scenarios(3_000_000, 500_000, "good")
        assert len(scenarios) == 3

    def test_scenario_keys(self, svc):
        scenarios = svc.generate_alternative_scenarios(3_000_000, 500_000, "good")
        for s in scenarios:
            assert "scenario" in s
            assert "description" in s
            assert "estimated_value" in s

    def test_sell_as_is_floor(self, svc):
        scenarios = svc.generate_alternative_scenarios(100_000, 500_000, "poor")
        sell_as_is = scenarios[0]
        assert sell_as_is["estimated_value"] >= 100_000

    def test_scrap_floor(self, svc):
        scenarios = svc.generate_alternative_scenarios(100_000, 500_000, "poor")
        scrap = scenarios[2]
        assert scrap["estimated_value"] >= 50_000

    def test_repair_then_sell_net(self, svc):
        scenarios = svc.generate_alternative_scenarios(3_000_000, 500_000, "good")
        repair_sell = scenarios[1]
        assert repair_sell["estimated_value"] == 3_000_000 - 500_000


# ---------------------------------------------------------------------------
# calculate() async method
# ---------------------------------------------------------------------------


class TestCalculateAsync:
    @pytest.mark.asyncio
    async def test_with_explicit_repair_cost(self, svc):
        result = await svc.calculate(
            vehicle_make="VW",
            vehicle_model="Golf",
            vehicle_year=2020,
            mileage_km=80_000,
            condition="good",
            repair_cost_huf=500_000,
        )
        assert result["currency"] == "HUF"
        assert result["confidence_score"] == 0.80
        assert result["repair_cost_min"] == int(500_000 * 0.85)
        assert result["repair_cost_max"] == int(500_000 * 1.15)
        assert result["breakdown"]["parts_cost"] == int(500_000 * 0.60)
        assert result["breakdown"]["labor_cost"] == int(500_000 * 0.40)
        assert result["recommendation"] in ("repair", "sell", "scrap")

    @pytest.mark.asyncio
    async def test_no_cost_info_fallback(self, svc):
        result = await svc.calculate(
            vehicle_make="TOYOTA",
            vehicle_model="Corolla",
            vehicle_year=2018,
            mileage_km=120_000,
            condition="fair",
        )
        assert result["repair_cost_min"] == 50_000
        assert result["repair_cost_max"] == 300_000
        assert result["confidence_score"] == 0.40

    @pytest.mark.asyncio
    async def test_with_fuel_type_adjustment(self, svc):
        result_benzin = await svc.calculate(
            vehicle_make="VW",
            vehicle_model="Golf",
            vehicle_year=2020,
            mileage_km=50_000,
            condition="good",
            repair_cost_huf=200_000,
            fuel_type="benzin",
        )
        result_hybrid = await svc.calculate(
            vehicle_make="VW",
            vehicle_model="Golf",
            vehicle_year=2020,
            mileage_km=50_000,
            condition="good",
            repair_cost_huf=200_000,
            fuel_type="hybrid",
        )
        assert result_hybrid["vehicle_value_avg"] > result_benzin["vehicle_value_avg"]

    @pytest.mark.asyncio
    async def test_with_diagnosis_id_found(self, svc):
        cost_data = {
            "total_min": 100_000,
            "total_max": 300_000,
            "parts_avg": 120_000,
            "labor_avg": 80_000,
        }
        with patch.object(
            svc, "_fetch_diagnosis_cost", new_callable=AsyncMock, return_value=cost_data
        ):
            result = await svc.calculate(
                vehicle_make="VW",
                vehicle_model="Golf",
                vehicle_year=2020,
                mileage_km=80_000,
                condition="good",
                diagnosis_id="some-uuid",
            )
        assert result["repair_cost_min"] == 100_000
        assert result["repair_cost_max"] == 300_000
        assert result["confidence_score"] == 0.75

    @pytest.mark.asyncio
    async def test_with_diagnosis_id_not_found(self, svc):
        with patch.object(svc, "_fetch_diagnosis_cost", new_callable=AsyncMock, return_value=None):
            result = await svc.calculate(
                vehicle_make="VW",
                vehicle_model="Golf",
                vehicle_year=2020,
                mileage_km=80_000,
                condition="good",
                diagnosis_id="missing-uuid",
            )
        assert result["repair_cost_min"] == 50_000
        assert result["confidence_score"] == 0.40

    @pytest.mark.asyncio
    async def test_result_has_all_keys(self, svc):
        result = await svc.calculate(
            vehicle_make="DACIA",
            vehicle_model="Sandero",
            vehicle_year=2019,
            mileage_km=70_000,
            condition="good",
            repair_cost_huf=200_000,
        )
        expected_keys = {
            "vehicle_value_min",
            "vehicle_value_max",
            "vehicle_value_avg",
            "repair_cost_min",
            "repair_cost_max",
            "ratio",
            "recommendation",
            "recommendation_text",
            "breakdown",
            "factors",
            "alternative_scenarios",
            "confidence_score",
            "currency",
        }
        assert expected_keys.issubset(result.keys())


# ---------------------------------------------------------------------------
# Static data integrity
# ---------------------------------------------------------------------------


class TestStaticData:
    def test_all_premium_makes_in_mapping(self):
        for make in PREMIUM_MAKES:
            assert make in MAKE_CLASS_MAPPING

    def test_condition_multipliers_ordered(self):
        assert CONDITION_MULTIPLIERS["excellent"] > CONDITION_MULTIPLIERS["good"]
        assert CONDITION_MULTIPLIERS["good"] > CONDITION_MULTIPLIERS["fair"]
        assert CONDITION_MULTIPLIERS["fair"] > CONDITION_MULTIPLIERS["poor"]

    def test_depreciation_decreasing(self):
        values = [STANDARD_DEPRECIATION[y] for y in sorted(STANDARD_DEPRECIATION)]
        for i in range(1, len(values)):
            assert values[i] < values[i - 1]
