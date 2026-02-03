"""
Neo4j graph database models using Neomodel.
"""

from neomodel import (
    ArrayProperty,
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
config.DATABASE_URL = f"bolt://{settings.NEO4J_USER}:{settings.NEO4J_PASSWORD}@{settings.NEO4J_URI.replace('bolt://', '')}"


# Relationship models
class CausesRel(StructuredRel):
    """Relationship: DTC causes Symptom."""

    confidence = FloatProperty(default=0.5)
    source = StringProperty()


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
    optional = StringProperty(default="false")


class HasCommonIssueRel(StructuredRel):
    """Relationship: Vehicle has common issue DTC."""

    frequency = StringProperty(default="common")  # rare, uncommon, common, very_common
    year_start = IntegerProperty()
    year_end = IntegerProperty()


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
    indicates_failure_of = RelationshipTo("ComponentNode", "INDICATES_FAILURE_OF", model=IndicatesFailureRel)
    related_to = RelationshipTo("DTCNode", "RELATED_TO")


class SymptomNode(StructuredNode):
    """Symptom node."""

    uid = UniqueIdProperty()
    name = StringProperty(required=True)
    description = StringProperty()
    description_hu = StringProperty()
    severity = StringProperty(default="medium")

    # Relationships
    caused_by = RelationshipFrom("DTCNode", "CAUSES", model=CausesRel)
    requires_check = RelationshipTo("TestPointNode", "REQUIRES_CHECK")


class ComponentNode(StructuredNode):
    """Vehicle component node."""

    uid = UniqueIdProperty()
    name = StringProperty(required=True)
    name_hu = StringProperty()
    system = StringProperty()  # engine, transmission, brakes, etc.
    part_number = StringProperty()

    # Relationships
    failure_indicated_by = RelationshipFrom("DTCNode", "INDICATES_FAILURE_OF", model=IndicatesFailureRel)
    repaired_by = RelationshipTo("RepairNode", "REPAIRED_BY", model=RepairedByRel)
    contains = RelationshipTo("ComponentNode", "CONTAINS")
    contained_in = RelationshipFrom("ComponentNode", "CONTAINS")


class RepairNode(StructuredNode):
    """Repair action node."""

    uid = UniqueIdProperty()
    name = StringProperty(required=True)
    description = StringProperty()
    description_hu = StringProperty()
    difficulty = StringProperty(default="intermediate")  # beginner, intermediate, advanced, professional
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
    name = StringProperty(required=True)
    name_hu = StringProperty()
    part_number = StringProperty()
    oem_part_number = StringProperty()
    price_min = IntegerProperty()
    price_max = IntegerProperty()
    currency = StringProperty(default="HUF")

    # Relationships
    used_in = RelationshipFrom("RepairNode", "USES_PART", model=UsesPartRel)


class TestPointNode(StructuredNode):
    """Diagnostic test point node."""

    uid = UniqueIdProperty()
    name = StringProperty(required=True)
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
    make = StringProperty(required=True)
    model = StringProperty(required=True)
    year_start = IntegerProperty()
    year_end = IntegerProperty()
    platform = StringProperty()
    engine_codes = ArrayProperty(StringProperty())

    # Relationships
    has_common_issue = RelationshipTo("DTCNode", "HAS_COMMON_ISSUE", model=HasCommonIssueRel)
    uses_component = RelationshipTo("ComponentNode", "USES_COMPONENT")


# Utility functions
async def create_dtc_symptom_relationship(
    dtc_code: str,
    symptom_name: str,
    confidence: float = 0.5,
) -> bool:
    """Create a relationship between a DTC and a symptom."""
    try:
        dtc = DTCNode.nodes.get_or_none(code=dtc_code)
        if not dtc:
            return False

        symptom = SymptomNode.nodes.get_or_none(name=symptom_name)
        if not symptom:
            symptom = SymptomNode(name=symptom_name).save()

        dtc.causes.connect(symptom, {"confidence": confidence})
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

    result = {
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
        result["symptoms"].append({
            "name": symptom.name,
            "description": symptom.description_hu or symptom.description,
            "confidence": rel.confidence,
        })

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
                repair_data["parts"].append({
                    "name": part.name_hu or part.name,
                    "part_number": part.part_number,
                    "price_range": f"{part.price_min}-{part.price_max} {part.currency}",
                    "quantity": part_rel.quantity,
                })

            comp_data["repairs"].append(repair_data)
            result["repairs"].append(repair_data)

        result["components"].append(comp_data)

    return result
