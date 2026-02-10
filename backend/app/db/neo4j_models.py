"""
Neo4j graph database models using Neomodel.
"""

import asyncio
from typing import Any, Dict, Optional

from neomodel import (
    ArrayProperty,
    BooleanProperty,
    FloatProperty,
    IntegerProperty,
    RelationshipFrom,
    RelationshipTo,
    StringProperty,
    StructuredNode,
    StructuredRel,
    UniqueIdProperty,
    config,
)

from app.core.config import settings

# Configure Neo4j connection
# Handle both bolt:// and neo4j+s:// URI schemes (Aura uses neo4j+s://)
_neo4j_host = settings.NEO4J_URI
for _scheme in ("neo4j+s://", "neo4j+ssc://", "neo4j://", "bolt+s://", "bolt://"):
    _neo4j_host = _neo4j_host.replace(_scheme, "")
config.DATABASE_URL = f"{settings.NEO4J_URI.split('://')[0]}://{settings.NEO4J_USER}:{settings.NEO4J_PASSWORD}@{_neo4j_host}"


# Relationship models
class CausesRel(StructuredRel):
    """Relationship: DTC causes Symptom."""

    confidence = FloatProperty(default=0.5)
    data_source = StringProperty()  # Renamed from 'source' - conflicts with neomodel internals


class IndicatesFailureRel(StructuredRel):
    """Relationship: DTC indicates failure of Component."""

    confidence = FloatProperty(default=0.5)
    failure_mode = StringProperty()


class RepairedByRel(StructuredRel):
    """Relationship: Component is repaired by Repair."""

    difficulty = StringProperty(default="intermediate")
    estimated_time_minutes = IntegerProperty()


class UsesPartRel(StructuredRel):
    """Relationship: Repair uses Part."""

    quantity = IntegerProperty(default=1)
    optional = BooleanProperty(default=False)


class HasCommonIssueRel(StructuredRel):
    """Relationship: Vehicle has common issue DTC."""

    frequency = StringProperty(default="common")  # rare, uncommon, common, very_common
    year_start = IntegerProperty()
    year_end = IntegerProperty()
    occurrence_count = IntegerProperty()
    data_source = StringProperty()  # nhtsa, tsb, forum


class RequiresRepairRel(StructuredRel):
    """Relationship: DTC requires Repair for specific vehicle."""

    confidence = FloatProperty(default=0.7)
    is_primary_fix = BooleanProperty(default=False)
    estimated_labor_hours = FloatProperty()


class UsesEngineRel(StructuredRel):
    """Relationship: Vehicle uses Engine."""

    year_start = IntegerProperty()
    year_end = IntegerProperty()
    is_base_engine = BooleanProperty(default=False)
    variant_name = StringProperty()  # e.g., "2.0 TSI 190HP"


class SharesPlatformRel(StructuredRel):
    """Relationship: Vehicle shares platform with another vehicle."""

    platform_code = StringProperty()


# Node models
class DTCNode(StructuredNode):
    """DTC (Diagnostic Trouble Code) node."""

    code = StringProperty(unique_index=True, required=True)
    description_en = StringProperty(required=True)
    description_hu = StringProperty()
    category = StringProperty(required=True)  # powertrain, body, chassis, network
    severity = StringProperty(default="medium")
    is_generic = StringProperty(default="true")
    system = StringProperty()

    # Relationships
    causes = RelationshipTo("SymptomNode", "CAUSES", model=CausesRel)
    indicates_failure_of = RelationshipTo(
        "ComponentNode", "INDICATES_FAILURE_OF", model=IndicatesFailureRel
    )
    related_to = RelationshipTo("DTCNode", "RELATED_TO")


class SymptomNode(StructuredNode):
    """Symptom node representing vehicle symptoms."""

    uid = UniqueIdProperty()
    symptom_id = StringProperty(index=True)  # e.g., "ENG001", "BRK002"
    name = StringProperty(required=True, index=True)  # Hungarian name (primary)
    name_en = StringProperty()  # English name
    description = StringProperty()  # English description
    description_hu = StringProperty()  # Hungarian description
    category = StringProperty(index=True)  # engine, transmission, brakes, etc.
    severity = StringProperty(default="medium")  # critical, high, medium, low
    keywords = ArrayProperty(StringProperty())  # Search keywords in Hungarian
    possible_causes = ArrayProperty(StringProperty())  # List of possible causes
    diagnostic_steps = ArrayProperty(StringProperty())  # Diagnostic procedures

    # Relationships
    caused_by = RelationshipFrom("DTCNode", "CAUSES", model=CausesRel)
    requires_check = RelationshipTo("TestPointNode", "REQUIRES_CHECK")
    related_symptoms = RelationshipTo("SymptomNode", "RELATED_TO")


class ComponentNode(StructuredNode):
    """Vehicle component node."""

    uid = UniqueIdProperty()
    name = StringProperty(required=True, index=True)
    name_hu = StringProperty()
    system = StringProperty(index=True)  # engine, transmission, brakes, etc.
    part_number = StringProperty()

    # Relationships
    failure_indicated_by = RelationshipFrom(
        "DTCNode", "INDICATES_FAILURE_OF", model=IndicatesFailureRel
    )
    repaired_by = RelationshipTo("RepairNode", "REPAIRED_BY", model=RepairedByRel)
    contains = RelationshipTo("ComponentNode", "CONTAINS")
    contained_in = RelationshipFrom("ComponentNode", "CONTAINS")


class RepairNode(StructuredNode):
    """Repair action node."""

    uid = UniqueIdProperty()
    name = StringProperty(required=True, index=True)
    description = StringProperty()
    description_hu = StringProperty()
    difficulty = StringProperty(
        default="intermediate"
    )  # beginner, intermediate, advanced, professional
    estimated_time_minutes = IntegerProperty()
    estimated_cost_min = IntegerProperty()
    estimated_cost_max = IntegerProperty()

    # Relationships
    repairs = RelationshipFrom("ComponentNode", "REPAIRED_BY", model=RepairedByRel)
    uses_parts = RelationshipTo("PartNode", "USES_PART", model=UsesPartRel)
    diagnostic_steps = ArrayProperty(StringProperty())


class PartNode(StructuredNode):
    """Replacement part node."""

    uid = UniqueIdProperty()
    name = StringProperty(required=True, index=True)
    name_hu = StringProperty()
    part_number = StringProperty(index=True)
    oem_part_number = StringProperty()
    price_min = IntegerProperty()
    price_max = IntegerProperty()
    currency = StringProperty(default="HUF")

    # Relationships
    used_in = RelationshipFrom("RepairNode", "USES_PART", model=UsesPartRel)


class TestPointNode(StructuredNode):
    """Diagnostic test point node."""

    uid = UniqueIdProperty()
    name = StringProperty(required=True, index=True)
    description = StringProperty()
    description_hu = StringProperty()
    test_type = StringProperty()  # visual, multimeter, scanner, oscilloscope
    expected_value = StringProperty()
    expected_range_min = FloatProperty()
    expected_range_max = FloatProperty()
    unit = StringProperty()

    # Relationships
    checks_for = RelationshipFrom("SymptomNode", "REQUIRES_CHECK")
    leads_to = RelationshipTo("RepairNode", "LEADS_TO")


class VehicleNode(StructuredNode):
    """Vehicle type node."""

    uid = UniqueIdProperty()
    make = StringProperty(required=True, index=True)
    model = StringProperty(required=True, index=True)
    year_start = IntegerProperty()
    year_end = IntegerProperty()
    platform = StringProperty(index=True)
    engine_codes = ArrayProperty(StringProperty())
    body_types = ArrayProperty(StringProperty())
    country = StringProperty()
    segment = StringProperty()  # A, B, C, D, E, F

    # Relationships
    has_common_issue = RelationshipTo("DTCNode", "HAS_COMMON_ISSUE", model=HasCommonIssueRel)
    uses_component = RelationshipTo("ComponentNode", "USES_COMPONENT")
    uses_engine = RelationshipTo("EngineNode", "USES_ENGINE", model=UsesEngineRel)
    common_repair = RelationshipTo("RepairNode", "COMMON_REPAIR", model=RequiresRepairRel)
    shares_platform_with = RelationshipTo("VehicleNode", "SHARES_PLATFORM", model=SharesPlatformRel)


class EngineNode(StructuredNode):
    """Engine specification node."""

    uid = UniqueIdProperty()
    code = StringProperty(unique_index=True, required=True)
    name = StringProperty(index=True)
    family = StringProperty(index=True)  # EA888, B58, M264, etc.
    manufacturer = StringProperty(index=True)

    # Specifications
    displacement_cc = IntegerProperty()
    displacement_l = FloatProperty()
    cylinders = IntegerProperty()
    configuration = StringProperty()  # inline, v, boxer, rotary, electric
    fuel_type = StringProperty(required=True, index=True)  # gasoline, diesel, hybrid, electric
    aspiration = StringProperty()  # naturally_aspirated, turbo, supercharged, electric

    # Power output
    power_hp = IntegerProperty()
    power_kw = IntegerProperty()
    torque_nm = IntegerProperty()

    # Production years
    year_start = IntegerProperty()
    year_end = IntegerProperty()

    # Relationships
    used_in = RelationshipFrom("VehicleNode", "USES_ENGINE", model=UsesEngineRel)
    common_issues = RelationshipTo("DTCNode", "HAS_COMMON_ISSUE", model=HasCommonIssueRel)
    requires_repair = RelationshipTo("RepairNode", "COMMON_REPAIR", model=RequiresRepairRel)


class PlatformNode(StructuredNode):
    """Vehicle platform node."""

    uid = UniqueIdProperty()
    code = StringProperty(unique_index=True, required=True)
    name = StringProperty(required=True, index=True)
    manufacturer = StringProperty(index=True)
    segment = StringProperty()  # A, B, C, D, E, F
    year_start = IntegerProperty()
    year_end = IntegerProperty()
    drivetrain_options = ArrayProperty(StringProperty())  # FWD, RWD, AWD

    # Relationships
    vehicles = RelationshipFrom("VehicleNode", "SHARES_PLATFORM", model=SharesPlatformRel)


# Utility functions
async def create_dtc_symptom_relationship(
    dtc_code: str,
    symptom_name: str,
    confidence: float = 0.5,
) -> bool:
    """Create a relationship between a DTC and a symptom."""
    try:
        dtc = await asyncio.to_thread(DTCNode.nodes.get_or_none, code=dtc_code)
        if not dtc:
            return False

        symptom = await asyncio.to_thread(SymptomNode.nodes.get_or_none, name=symptom_name)
        if not symptom:
            symptom = await asyncio.to_thread(lambda: SymptomNode(name=symptom_name).save())

        await asyncio.to_thread(dtc.causes.connect, symptom, {"confidence": confidence})
        return True
    except Exception:
        return False


async def get_diagnostic_path(dtc_code: str) -> dict:
    """
    Get the full diagnostic path for a DTC code.

    Returns a graph of related symptoms, components, repairs, and parts.
    """
    dtc = DTCNode.nodes.get_or_none(code=dtc_code)
    if not dtc:
        return {}

    result: Dict[str, Any] = {
        "dtc": {
            "code": dtc.code,
            "description": dtc.description_hu or dtc.description_en,
            "severity": dtc.severity,
        },
        "symptoms": [],
        "components": [],
        "repairs": [],
    }

    # Get symptoms
    for symptom in dtc.causes.all():
        rel = dtc.causes.relationship(symptom)
        result["symptoms"].append(
            {
                "name": symptom.name,
                "description": symptom.description_hu or symptom.description,
                "confidence": rel.confidence,
            }
        )

    # Get components
    for component in dtc.indicates_failure_of.all():
        rel = dtc.indicates_failure_of.relationship(component)
        comp_data = {
            "name": component.name_hu or component.name,
            "system": component.system,
            "failure_mode": rel.failure_mode,
            "repairs": [],
        }

        # Get repairs for this component
        for repair in component.repaired_by.all():
            repair_rel = component.repaired_by.relationship(repair)
            repair_data = {
                "name": repair.name,
                "description": repair.description_hu or repair.description,
                "difficulty": repair_rel.difficulty,
                "estimated_time_minutes": repair_rel.estimated_time_minutes,
                "estimated_cost_min": repair.estimated_cost_min,
                "estimated_cost_max": repair.estimated_cost_max,
                "parts": [],
            }

            # Get parts for this repair
            for part in repair.uses_parts.all():
                part_rel = repair.uses_parts.relationship(part)
                repair_data["parts"].append(
                    {
                        "name": part.name_hu or part.name,
                        "part_number": part.part_number,
                        "price_range": f"{part.price_min}-{part.price_max} {part.currency}",
                        "quantity": part_rel.quantity,
                    }
                )

            comp_data["repairs"].append(repair_data)
            result["repairs"].append(repair_data)

        result["components"].append(comp_data)

    return result


async def get_vehicle_common_issues(make: str, model: str, year: Optional[int] = None) -> dict:
    """
    Get common DTC codes and issues for a specific vehicle.

    Args:
        make: Vehicle manufacturer ID (e.g., 'volkswagen')
        model: Vehicle model name (e.g., 'Golf VIII')
        year: Optional model year for filtering

    Returns:
        Dictionary with common DTCs, repairs, and components for the vehicle.
    """
    vehicle = VehicleNode.nodes.filter(make=make, model=model).first_or_none()
    if not vehicle:
        return {"vehicle": None, "common_dtcs": [], "common_repairs": []}

    result: Dict[str, Any] = {
        "vehicle": {
            "make": vehicle.make,
            "model": vehicle.model,
            "year_start": vehicle.year_start,
            "year_end": vehicle.year_end,
            "platform": vehicle.platform,
            "engine_codes": vehicle.engine_codes or [],
        },
        "common_dtcs": [],
        "common_repairs": [],
    }

    # Get common DTC issues
    for dtc in vehicle.has_common_issue.all():
        rel = vehicle.has_common_issue.relationship(dtc)

        # Filter by year if specified
        if year:
            rel_year_start = rel.year_start or vehicle.year_start
            rel_year_end = rel.year_end or vehicle.year_end or 2030
            if not (rel_year_start <= year <= rel_year_end):
                continue

        dtc_data = {
            "code": dtc.code,
            "description": dtc.description_hu or dtc.description_en,
            "severity": dtc.severity,
            "frequency": rel.frequency,
            "occurrence_count": rel.occurrence_count,
        }

        # Get repair recommendations for this DTC
        dtc_data["recommended_repairs"] = []
        for component in dtc.indicates_failure_of.all():
            for repair in component.repaired_by.all():
                component.repaired_by.relationship(repair)
                dtc_data["recommended_repairs"].append(
                    {
                        "name": repair.description_hu or repair.name,
                        "difficulty": repair.difficulty,
                        "estimated_time_minutes": repair.estimated_time_minutes,
                        "estimated_cost": f"{repair.estimated_cost_min}-{repair.estimated_cost_max} HUF",
                    }
                )

        result["common_dtcs"].append(dtc_data)

    # Get common repairs directly linked to vehicle
    for repair in vehicle.common_repair.all():
        rel = vehicle.common_repair.relationship(repair)
        result["common_repairs"].append(
            {
                "name": repair.description_hu or repair.name,
                "difficulty": repair.difficulty,
                "confidence": rel.confidence,
                "is_primary_fix": bool(rel.is_primary_fix),
                "estimated_time_minutes": repair.estimated_time_minutes,
                "estimated_cost": f"{repair.estimated_cost_min}-{repair.estimated_cost_max} HUF",
            }
        )

    return result


async def get_engine_common_issues(engine_code: str) -> dict:
    """
    Get common issues for a specific engine code.

    Args:
        engine_code: Engine code (e.g., 'EA888_GEN3_2.0_190')

    Returns:
        Dictionary with engine info and common DTCs.
    """
    engine = EngineNode.nodes.get_or_none(code=engine_code)
    if not engine:
        return {"engine": None, "common_dtcs": [], "vehicles_using": []}

    result: Dict[str, Any] = {
        "engine": {
            "code": engine.code,
            "name": engine.name,
            "family": engine.family,
            "manufacturer": engine.manufacturer,
            "displacement_l": engine.displacement_l,
            "fuel_type": engine.fuel_type,
            "power_hp": engine.power_hp,
            "torque_nm": engine.torque_nm,
        },
        "common_dtcs": [],
        "vehicles_using": [],
    }

    # Get common DTC issues
    for dtc in engine.common_issues.all():
        rel = engine.common_issues.relationship(dtc)
        result["common_dtcs"].append(
            {
                "code": dtc.code,
                "description": dtc.description_hu or dtc.description_en,
                "severity": dtc.severity,
                "frequency": rel.frequency,
            }
        )

    # Get vehicles using this engine
    for vehicle in engine.used_in.all():
        rel = engine.used_in.relationship(vehicle)
        result["vehicles_using"].append(
            {
                "make": vehicle.make,
                "model": vehicle.model,
                "year_start": rel.year_start or vehicle.year_start,
                "year_end": rel.year_end or vehicle.year_end,
                "variant": rel.variant_name,
            }
        )

    return result


async def find_similar_vehicles(make: str, model: str) -> list:
    """
    Find vehicles that share the same platform (likely similar issues).

    Args:
        make: Vehicle manufacturer ID
        model: Vehicle model name

    Returns:
        List of similar vehicles sharing the same platform.
    """
    vehicle = VehicleNode.nodes.filter(make=make, model=model).first_or_none()
    if not vehicle or not vehicle.platform:
        return []

    similar = []
    for other_vehicle in vehicle.shares_platform_with.all():
        rel = vehicle.shares_platform_with.relationship(other_vehicle)
        similar.append(
            {
                "make": other_vehicle.make,
                "model": other_vehicle.model,
                "platform": rel.platform_code,
                "year_start": other_vehicle.year_start,
                "year_end": other_vehicle.year_end,
            }
        )

    return similar
