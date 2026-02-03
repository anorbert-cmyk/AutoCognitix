"""
DTC (Diagnostic Trouble Code) endpoints.
"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, status

from app.api.v1.schemas.dtc import (
    DTCCode,
    DTCCodeDetail,
    DTCSearchResult,
    DTCCategory,
)

router = APIRouter()


@router.get("/search", response_model=List[DTCSearchResult])
async def search_dtc_codes(
    q: str = Query(..., min_length=1, description="Search query (code or description)"),
    category: Optional[DTCCategory] = Query(None, description="Filter by category"),
    make: Optional[str] = Query(None, description="Filter by vehicle make"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results to return"),
):
    """
    Search for DTC codes by code or description.

    Supports both generic OBD-II codes and manufacturer-specific codes.
    Hungarian descriptions are included where available.

    Args:
        q: Search query
        category: Optional category filter (powertrain, body, chassis, network)
        make: Optional vehicle make filter for manufacturer-specific codes
        limit: Maximum number of results

    Returns:
        List of matching DTC codes
    """
    # TODO: Implement with database and vector search
    # Placeholder data

    query = q.upper()

    # Sample DTC codes for demonstration
    sample_codes = [
        DTCSearchResult(
            code="P0101",
            description_en="Mass Air Flow Circuit Range/Performance",
            description_hu="Levegőtömeg-mérő áramkör tartomány/teljesítmény hiba",
            category="powertrain",
            severity="medium",
            is_generic=True,
        ),
        DTCSearchResult(
            code="P0171",
            description_en="System Too Lean (Bank 1)",
            description_hu="Rendszer túl sovány (Bank 1)",
            category="powertrain",
            severity="medium",
            is_generic=True,
        ),
        DTCSearchResult(
            code="P0300",
            description_en="Random/Multiple Cylinder Misfire Detected",
            description_hu="Véletlenszerű/többszörös hengerbedurranás észlelve",
            category="powertrain",
            severity="high",
            is_generic=True,
        ),
        DTCSearchResult(
            code="P0420",
            description_en="Catalyst System Efficiency Below Threshold (Bank 1)",
            description_hu="Katalizátor rendszer hatékonysága küszöb alatt (Bank 1)",
            category="powertrain",
            severity="medium",
            is_generic=True,
        ),
        DTCSearchResult(
            code="P0507",
            description_en="Idle Air Control System RPM Higher Than Expected",
            description_hu="Alapjárati levegő szabályozó rendszer - fordulatszám magasabb a vártnál",
            category="powertrain",
            severity="low",
            is_generic=True,
        ),
    ]

    # Filter by query
    results = [
        code for code in sample_codes
        if query in code.code or query.lower() in code.description_en.lower()
        or query.lower() in code.description_hu.lower()
    ]

    # Filter by category
    if category:
        results = [r for r in results if r.category == category.value]

    return results[:limit]


@router.get("/{code}", response_model=DTCCodeDetail)
async def get_dtc_code_detail(code: str):
    """
    Get detailed information about a specific DTC code.

    Includes symptoms, causes, diagnostic steps, and related codes.

    Args:
        code: The DTC code (e.g., P0101)

    Returns:
        Detailed DTC information
    """
    code = code.upper().strip()

    # Validate code format
    if not (len(code) == 5 and code[0] in "PBCU" and code[1:].isdigit()):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid DTC code format. Expected format: P0101, B1234, C0567, U0100",
        )

    # TODO: Implement with database lookup
    # Placeholder for P0101

    if code == "P0101":
        return DTCCodeDetail(
            code="P0101",
            description_en="Mass Air Flow Circuit Range/Performance",
            description_hu="Levegőtömeg-mérő áramkör tartomány/teljesítmény hiba",
            category="powertrain",
            severity="medium",
            is_generic=True,
            system="Fuel and Air Metering",
            symptoms=[
                "Motor teljesítményvesztése",
                "Egyenetlen alapjárat",
                "Nehéz indítás",
                "Megnövekedett üzemanyag-fogyasztás",
                "Fekete füst a kipufogóból",
            ],
            possible_causes=[
                "Szennyezett MAF szenzor",
                "Levegőszűrő eltömődése",
                "Vákuumszivárgás a szívórendszerben",
                "MAF szenzor meghibásodása",
                "Vezeték vagy csatlakozó probléma",
            ],
            diagnostic_steps=[
                "1. Vizuálisan ellenőrizze a MAF szenzort és a levegőszűrőt",
                "2. Ellenőrizze a MAF szenzor csatlakozóját és vezetékeit",
                "3. Tisztítsa meg a MAF szenzort speciális tisztítóval",
                "4. Ellenőrizze a szívórendszert vákuumszivárgás szempontjából",
                "5. Tesztelje a MAF szenzor jelét oszcilloszkóppal vagy multiméterecel",
            ],
            related_codes=["P0100", "P0102", "P0103", "P0171", "P0174"],
            common_vehicles=[
                "Volkswagen Golf (2010-2020)",
                "Audi A3 (2012-2020)",
                "Ford Focus (2011-2018)",
                "Toyota Corolla (2014-2019)",
            ],
        )

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"DTC code {code} not found",
    )


@router.get("/{code}/related", response_model=List[DTCSearchResult])
async def get_related_codes(code: str):
    """
    Get DTC codes related to the specified code.

    Related codes are determined by:
    - Same system/component
    - Often appear together
    - Similar root causes

    Args:
        code: The DTC code

    Returns:
        List of related DTC codes
    """
    code = code.upper().strip()

    # TODO: Implement with Neo4j graph queries
    # Placeholder

    return [
        DTCSearchResult(
            code="P0100",
            description_en="Mass Air Flow Circuit Malfunction",
            description_hu="Levegőtömeg-mérő áramkör meghibásodás",
            category="powertrain",
            severity="medium",
            is_generic=True,
        ),
        DTCSearchResult(
            code="P0102",
            description_en="Mass Air Flow Circuit Low Input",
            description_hu="Levegőtömeg-mérő áramkör alacsony bemenet",
            category="powertrain",
            severity="medium",
            is_generic=True,
        ),
        DTCSearchResult(
            code="P0103",
            description_en="Mass Air Flow Circuit High Input",
            description_hu="Levegőtömeg-mérő áramkör magas bemenet",
            category="powertrain",
            severity="medium",
            is_generic=True,
        ),
    ]


@router.get("/categories/list", response_model=List[dict])
async def get_dtc_categories():
    """
    Get list of DTC categories with descriptions.

    Returns:
        List of categories with codes and descriptions
    """
    return [
        {
            "code": "P",
            "name": "Powertrain",
            "name_hu": "Hajtáslánc",
            "description": "Engine, transmission, and emission systems",
            "description_hu": "Motor, váltó és emissziós rendszerek",
        },
        {
            "code": "B",
            "name": "Body",
            "name_hu": "Karosszéria",
            "description": "Body systems including airbags, A/C, lighting",
            "description_hu": "Karosszéria rendszerek: légzsákok, klíma, világítás",
        },
        {
            "code": "C",
            "name": "Chassis",
            "name_hu": "Alváz",
            "description": "Chassis systems including ABS, steering, suspension",
            "description_hu": "Alváz rendszerek: ABS, kormányzás, felfüggesztés",
        },
        {
            "code": "U",
            "name": "Network",
            "name_hu": "Hálózat",
            "description": "Communication network and module systems",
            "description_hu": "Kommunikációs hálózat és vezérlő modulok",
        },
    ]
