"""
DTC (Diagnostic Trouble Code) endpoints.

Provides full CRUD operations for DTC codes with:
- PostgreSQL for structured data
- Neo4j for relationship graphs (symptoms, components, repairs)
- Qdrant for semantic search
- Redis caching for performance optimization

Performance optimizations:
- Redis caching for DTC lookups (1 hour TTL)
- Redis caching for search results (15 min TTL)
- Async embedding generation
- Response compression via middleware
"""

import logging
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.schemas.dtc import (
    DTCBulkImport,
    DTCCategory,
    DTCCode,
    DTCCodeDetail,
    DTCCreate,
    DTCSearchResult,
)
from app.core.log_sanitizer import sanitize_log
from app.db.postgres.models import DTCCode as DTCCodeModel
from app.db.postgres.repositories import DTCCodeRepository
from app.db.postgres.session import get_db
from app.db.qdrant_client import qdrant_client
from app.services.embedding_service import get_embedding_service

router = APIRouter()
logger = logging.getLogger(__name__)


# =============================================================================
# OpenAPI Response Examples
# =============================================================================

SEARCH_RESPONSES: Dict[Union[int, str], Dict[str, Any]] = {
    200: {
        "description": "DTC codes matching search criteria",
        "content": {
            "application/json": {
                "example": [
                    {
                        "code": "P0101",
                        "description_en": "Mass Air Flow Circuit Range/Performance",
                        "description_hu": "Levegotomeg-mero aramkor tartomany/teljesitmeny hiba",
                        "category": "powertrain",
                        "is_generic": True,
                        "severity": "medium",
                        "relevance_score": 0.95,
                    },
                    {
                        "code": "P0100",
                        "description_en": "Mass Air Flow Circuit Malfunction",
                        "description_hu": "Levegotomeg-mero aramkor meghibasodas",
                        "category": "powertrain",
                        "is_generic": True,
                        "severity": "medium",
                        "relevance_score": 0.82,
                    },
                ]
            }
        },
    }
}

DTC_DETAIL_RESPONSES: Dict[Union[int, str], Dict[str, Any]] = {
    200: {
        "description": "Detailed DTC code information",
        "content": {
            "application/json": {
                "example": {
                    "code": "P0101",
                    "description_en": "Mass Air Flow Circuit Range/Performance",
                    "description_hu": "Levegotomeg-mero aramkor tartomany/teljesitmeny hiba",
                    "category": "powertrain",
                    "is_generic": True,
                    "severity": "medium",
                    "system": "Fuel and Air Metering",
                    "symptoms": [
                        "Motor teljesitmenyvesztese",
                        "Egyenetlen alapjarat",
                        "Nehez inditas",
                    ],
                    "possible_causes": [
                        "Szennyezett MAF szenzor",
                        "Levegoszuro eltomodes",
                        "Vakuumszivarga",
                    ],
                    "diagnostic_steps": [
                        "Vizualisan ellenorizze a MAF szenzort",
                        "Tisztitsa meg a MAF szenzort specialis tisztitoval",
                        "Ellenorizze a levegoszurot",
                    ],
                    "related_codes": ["P0100", "P0102", "P0103"],
                    "common_vehicles": [],
                    "manufacturer_code": None,
                }
            }
        },
    },
    400: {
        "description": "Invalid DTC code format",
        "content": {
            "application/json": {
                "example": {
                    "detail": "Invalid DTC code format. Expected format: P0101, B1234, C0567, U0100"
                }
            }
        },
    },
    404: {
        "description": "DTC code not found",
        "content": {
            "application/json": {"example": {"detail": "DTC code P9999 not found in database"}}
        },
    },
}

RELATED_CODES_RESPONSES: Dict[Union[int, str], Dict[str, Any]] = {
    200: {
        "description": "Related DTC codes",
        "content": {
            "application/json": {
                "example": [
                    {
                        "code": "P0100",
                        "description_en": "Mass Air Flow Circuit Malfunction",
                        "description_hu": "Levegotomeg-mero aramkor meghibasodas",
                        "category": "powertrain",
                        "is_generic": True,
                        "severity": "medium",
                        "relevance_score": 0.9,
                    }
                ]
            }
        },
    },
    404: {
        "description": "DTC code not found",
        "content": {"application/json": {"example": {"detail": "DTC code P9999 not found"}}},
    },
}

CATEGORIES_RESPONSES: Dict[Union[int, str], Dict[str, Any]] = {
    200: {
        "description": "List of DTC categories",
        "content": {
            "application/json": {
                "example": [
                    {
                        "code": "P",
                        "name": "Powertrain",
                        "name_hu": "Hajtaslánc",
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
                ]
            }
        },
    }
}

CREATE_DTC_RESPONSES: Dict[Union[int, str], Dict[str, Any]] = {
    201: {
        "description": "DTC code created successfully",
        "content": {
            "application/json": {
                "example": {
                    "code": "P0101",
                    "description_en": "Mass Air Flow Circuit Range/Performance",
                    "description_hu": "Levegotomeg-mero aramkor tartomany/teljesitmeny hiba",
                    "category": "powertrain",
                    "is_generic": True,
                }
            }
        },
    },
    400: {
        "description": "DTC code already exists",
        "content": {"application/json": {"example": {"detail": "DTC code P0101 already exists"}}},
    },
}

BULK_IMPORT_RESPONSES: Dict[Union[int, str], Dict[str, Any]] = {
    201: {
        "description": "Bulk import completed",
        "content": {
            "application/json": {
                "example": {"created": 10, "updated": 5, "skipped": 2, "errors": [], "total": 17}
            }
        },
    }
}


# =============================================================================
# Redis Cache Helpers
# =============================================================================


async def _get_cache_service():
    """Get Redis cache service, handling import errors gracefully."""
    try:
        from app.db.redis_cache import get_cache_service

        return await get_cache_service()
    except Exception as e:
        logger.debug(f"Cache service unavailable: {e}")
        return None


async def _cache_dtc_detail(code: str, detail_dict: dict) -> None:
    """Cache DTC detail in Redis."""
    cache = await _get_cache_service()
    if cache:
        await cache.set_dtc_code(code, detail_dict)


async def _get_cached_dtc_detail(code: str) -> Any:
    """Get cached DTC detail from Redis."""
    cache = await _get_cache_service()
    if cache:
        return await cache.get_dtc_code(code)
    return None


async def _cache_search_results(
    query: str,
    results: List[dict],
    category: Optional[str] = None,
    limit: int = 20,
) -> None:
    """Cache search results in Redis."""
    cache = await _get_cache_service()
    if cache:
        await cache.set_dtc_search_results(query, results, category, limit)


async def _get_cached_search_results(
    query: str,
    category: Optional[str] = None,
    limit: int = 20,
) -> Any:
    """Get cached search results from Redis."""
    cache = await _get_cache_service()
    if cache:
        return await cache.get_dtc_search_results(query, category, limit)
    return None


async def _invalidate_dtc_cache(code: str) -> None:
    """Invalidate cache for a specific DTC code."""
    cache = await _get_cache_service()
    if cache:
        await cache.delete(f"dtc:code:{code.upper()}")
        # Also invalidate related search caches
        await cache.delete_pattern("dtc:search:*")


async def _get_neo4j_relationships(code: str) -> Dict[str, Any]:
    """
    Get DTC relationships from Neo4j graph database.

    Args:
        code: DTC code to look up

    Returns:
        Dictionary with symptoms, components, repairs, and related codes from graph
    """
    try:
        from app.db.neo4j_models import get_diagnostic_path

        return await get_diagnostic_path(code)
    except ImportError:
        logger.warning("Neo4j models not available")
        return {}
    except Exception as e:
        logger.warning(
            f"Error fetching Neo4j relationships for {sanitize_log(code)}: {sanitize_log(str(e))}"
        )
        return {}


def _dtc_model_to_search_result(
    dtc: DTCCodeModel, relevance_score: Optional[float] = None
) -> DTCSearchResult:
    """Convert PostgreSQL model to API schema."""
    return DTCSearchResult(
        code=dtc.code,
        description_en=dtc.description_en,
        description_hu=dtc.description_hu,
        category=dtc.category,
        is_generic=dtc.is_generic,
        severity=dtc.severity,
        relevance_score=relevance_score,
    )


def _dtc_model_to_detail(
    dtc: DTCCodeModel, neo4j_data: Optional[Dict[str, Any]] = None
) -> DTCCodeDetail:
    """Convert PostgreSQL model to detailed API schema, enriching with Neo4j data."""
    # Start with PostgreSQL data
    detail = DTCCodeDetail(
        code=dtc.code,
        description_en=dtc.description_en,
        description_hu=dtc.description_hu,
        category=dtc.category,
        is_generic=dtc.is_generic,
        severity=dtc.severity,
        system=dtc.system,
        symptoms=dtc.symptoms or [],
        possible_causes=dtc.possible_causes or [],
        diagnostic_steps=dtc.diagnostic_steps or [],
        related_codes=dtc.related_codes or [],
        common_vehicles=[],  # Not stored in PostgreSQL
        manufacturer_code=dtc.manufacturer_code,
        freeze_frame_data=None,
    )

    # Enrich with Neo4j graph data if available
    if neo4j_data:
        # Add symptoms from graph (deduplicated)
        graph_symptoms = [s.get("name", "") for s in neo4j_data.get("symptoms", [])]
        all_symptoms = list(set(detail.symptoms + graph_symptoms))
        detail.symptoms = all_symptoms[:20]  # Limit to 20 symptoms

        # Add components as causes
        for comp in neo4j_data.get("components", []):
            cause = f"{comp.get('name', 'Unknown')}"
            if comp.get("failure_mode"):
                cause += f" - {comp['failure_mode']}"
            if cause not in detail.possible_causes:
                detail.possible_causes.append(cause)

        # Add repairs to diagnostic steps
        for repair in neo4j_data.get("repairs", []):
            step = repair.get("description", repair.get("name", ""))
            if step and step not in detail.diagnostic_steps:
                detail.diagnostic_steps.append(step)

    return detail


@router.get(
    "/search",
    response_model=List[DTCSearchResult],
    responses=SEARCH_RESPONSES,
    summary="Search DTC codes",
    description="""
**Search for DTC codes** by code or description text.

Supports multiple search modes:
- **Code search**: Partial code matching (e.g., "P01" matches P0100, P0101, etc.)
- **Text search**: Searches English and Hungarian descriptions
- **Semantic search**: AI-powered similarity search using Hungarian huBERT embeddings

**Search examples:**
- `q=P0101` - Find exact code match
- `q=MAF` - Find codes related to MAF sensor
- `q=motor nehezen indul` - Semantic search in Hungarian

**Performance:**
- Results cached in Redis for 15 minutes
- Use `skip_cache=true` for fresh results
    """,
)
async def search_dtc_codes(
    q: str = Query(..., min_length=1, description="Search query (code or description)"),
    category: Optional[DTCCategory] = Query(None, description="Filter by category"),
    make: Optional[str] = Query(None, description="Filter by vehicle make"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results to return"),
    use_semantic: bool = Query(True, description="Use semantic search (slower but better)"),
    skip_cache: bool = Query(False, description="Skip cache lookup"),
    db: AsyncSession = Depends(get_db),
):
    """
    Search for DTC codes by code or description.

    Supports both generic OBD-II codes and manufacturer-specific codes.
    Hungarian descriptions are included where available.

    Search modes:
    - Text search: Fast, matches code and description text
    - Semantic search: Uses AI embeddings for better Hungarian understanding

    Performance:
    - Results are cached in Redis for 15 minutes
    - Use skip_cache=true to force fresh results

    Args:
        q: Search query
        category: Optional category filter (powertrain, body, chassis, network)
        make: Optional vehicle make filter for manufacturer-specific codes
        limit: Maximum number of results
        use_semantic: Whether to use semantic search (default: True)
        skip_cache: Skip cache lookup (default: False)
        db: Database session

    Returns:
        List of matching DTC codes with relevance scores
    """
    repository = DTCCodeRepository(db)
    query = q.strip()
    category_filter = category.value if category else None

    # Check cache first (unless skip_cache is True)
    if not skip_cache:
        cached = await _get_cached_search_results(query, category_filter, limit)
        if cached:
            logger.debug(f"Cache HIT for search: {sanitize_log(query)}")
            return [DTCSearchResult(**item) for item in cached]

    # Check if query looks like a DTC code (starts with P, B, C, or U)
    is_code_query = (
        len(query) >= 1
        and query[0].upper() in "PBCU"
        and (len(query) == 1 or query[1:2].isdigit() or query[1:].upper() == query[1:])
    )

    results: List[DTCSearchResult] = []

    # If it looks like a code, prioritize exact matches
    if is_code_query:
        code_query = query.upper()

        # Try exact match first
        exact_match = await repository.get_by_code(code_query)
        if exact_match:
            results.append(_dtc_model_to_search_result(exact_match, relevance_score=1.0))

    # Text search from PostgreSQL
    category_filter = category.value if category else None
    text_results = await repository.search(query, category=category_filter, limit=limit)

    # Add text results (avoid duplicates)
    existing_codes = {r.code for r in results}
    for dtc in text_results:
        if dtc.code not in existing_codes:
            # Calculate simple relevance score
            relevance = 0.5
            query_lower = query.lower()
            if query_lower in dtc.code.lower():
                relevance = 0.9
            elif dtc.description_hu and query_lower in dtc.description_hu.lower():
                relevance = 0.7
            elif query_lower in dtc.description_en.lower():
                relevance = 0.6

            results.append(_dtc_model_to_search_result(dtc, relevance_score=relevance))
            existing_codes.add(dtc.code)

    # Semantic search if enabled and we have Hungarian text
    if use_semantic and not is_code_query and len(results) < limit:
        try:
            # Generate embedding for query
            embedding_service = get_embedding_service()
            query_embedding = embedding_service.embed_text(query, preprocess=True)

            # Search Qdrant
            semantic_results = await qdrant_client.search_dtc(
                query_vector=query_embedding,
                limit=limit,
                category=category_filter,
            )

            # Merge semantic results
            for result in semantic_results:
                payload = result.get("payload", {})
                code = payload.get("code", "")

                if code and code not in existing_codes:
                    # Fetch full details from PostgreSQL
                    semantic_dtc = await repository.get_by_code(code)
                    if semantic_dtc:
                        results.append(
                            _dtc_model_to_search_result(
                                semantic_dtc, relevance_score=result.get("score", 0.5)
                            )
                        )
                        existing_codes.add(code)

        except Exception as e:
            logger.warning(f"Semantic search failed, using text results only: {e}")

    # Sort by relevance score
    results.sort(key=lambda x: x.relevance_score or 0, reverse=True)

    final_results = results[:limit]

    # Cache results
    if not skip_cache:
        cache_data = [r.model_dump() for r in final_results]
        await _cache_search_results(query, cache_data, category_filter, limit)
        logger.debug(f"Cached search results for: {sanitize_log(query)}")

    return final_results


@router.get(
    "/categories/list",
    response_model=List[Dict[str, str]],
    responses=CATEGORIES_RESPONSES,
    summary="Get DTC categories",
    description="""
**Get list of DTC categories** with descriptions in English and Hungarian.

DTC codes follow the OBD-II standard:
- **P** (Powertrain): Engine, transmission, emissions
- **B** (Body): Airbags, A/C, lighting, doors
- **C** (Chassis): ABS, steering, suspension
- **U** (Network): CAN bus, module communication
    """,
)
async def get_dtc_categories() -> List[Dict[str, str]]:
    """
    Get list of DTC categories with descriptions.

    Returns:
        List of categories with codes and descriptions in English and Hungarian
    """
    return [
        {
            "code": "P",
            "name": "Powertrain",
            "name_hu": "Hajtaslánc",
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


@router.get(
    "/{code}",
    response_model=DTCCodeDetail,
    responses=DTC_DETAIL_RESPONSES,
    summary="Get DTC code details",
    description="""
**Get detailed information** about a specific DTC code.

Aggregates data from multiple sources:
- **PostgreSQL**: Core DTC data with Hungarian translations
- **Neo4j**: Graph relationships (symptoms, components, repairs)

**Response includes:**
- Code description (EN/HU)
- Severity level
- System classification
- Common symptoms
- Possible causes
- Diagnostic steps
- Related codes

**Cache:** Results cached in Redis for 1 hour.
    """,
)
async def get_dtc_code_detail(
    code: str,
    include_graph: bool = Query(True, description="Include Neo4j graph relationships"),
    skip_cache: bool = Query(False, description="Skip cache lookup"),
    db: AsyncSession = Depends(get_db),
):
    """
    Get detailed information about a specific DTC code.

    Includes symptoms, causes, diagnostic steps, and related codes.
    Data is aggregated from PostgreSQL and Neo4j for comprehensive results.

    Performance:
    - Results are cached in Redis for 1 hour
    - Use skip_cache=true to force fresh results

    Args:
        code: The DTC code (e.g., P0101)
        include_graph: Whether to include Neo4j relationships
        skip_cache: Skip cache lookup (default: False)
        db: Database session

    Returns:
        Detailed DTC information with all relationships

    Raises:
        400: Invalid DTC code format
        404: DTC code not found
    """
    code = code.upper().strip()

    # Validate code format
    if not (len(code) >= 5 and code[0] in "PBCU"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid DTC code format. Expected format: P0101, B1234, C0567, U0100",
        )

    # Check cache first (unless skip_cache is True)
    cache_key = f"{code}:{include_graph}"
    if not skip_cache:
        cached = await _get_cached_dtc_detail(cache_key)
        if cached:
            logger.debug(f"Cache HIT for DTC detail: {sanitize_log(code)}")
            return DTCCodeDetail(**cached)

    # Fetch from PostgreSQL
    repository = DTCCodeRepository(db)
    dtc = await repository.get_by_code(code)

    if not dtc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"DTC code {code} not found in database",
        )

    # Fetch graph relationships if requested
    neo4j_data = None
    if include_graph:
        neo4j_data = await _get_neo4j_relationships(code)

    result = _dtc_model_to_detail(dtc, neo4j_data)

    # Cache the result
    if not skip_cache:
        await _cache_dtc_detail(cache_key, result.model_dump())
        logger.debug(f"Cached DTC detail: {sanitize_log(code)}")

    return result


@router.get(
    "/{code}/related",
    response_model=List[DTCSearchResult],
    responses=RELATED_CODES_RESPONSES,
    summary="Get related DTC codes",
    description="""
**Get DTC codes related** to the specified code.

Related codes are determined by:
- **Database relationships**: Stored related_codes field
- **Neo4j graph**: RELATED_TO relationships in knowledge graph
- **Same system**: Codes affecting the same vehicle system
- **Pattern matching**: Codes with similar prefixes (e.g., P01xx)
    """,
)
async def get_related_codes(
    code: str,
    limit: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
):
    """
    Get DTC codes related to the specified code.

    Related codes are determined by:
    - Same system/component (from database)
    - Graph relationships (from Neo4j)
    - Often appear together
    - Similar root causes

    Args:
        code: The DTC code
        limit: Maximum number of results
        db: Database session

    Returns:
        List of related DTC codes

    Raises:
        404: DTC code not found
    """
    code = code.upper().strip()
    repository = DTCCodeRepository(db)

    # Fetch the original DTC
    dtc = await repository.get_by_code(code)
    if not dtc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"DTC code {code} not found",
        )

    results: List[DTCSearchResult] = []
    existing_codes = {code}  # Exclude the original code

    # Get related codes from PostgreSQL (stored in related_codes field)
    if dtc.related_codes:
        related = await repository.get_related_codes(code)
        for r in related:
            if r.code not in existing_codes:
                results.append(_dtc_model_to_search_result(r, relevance_score=0.9))
                existing_codes.add(r.code)

    # Get related codes from Neo4j graph
    try:
        from app.db.neo4j_models import DTCNode

        dtc_node = DTCNode.nodes.get_or_none(code=code)
        if dtc_node:
            # Get directly related codes
            for related_node in dtc_node.related_to.all():
                if related_node.code not in existing_codes:
                    # Fetch from PostgreSQL for full details
                    related_dtc = await repository.get_by_code(related_node.code)
                    if related_dtc:
                        results.append(
                            _dtc_model_to_search_result(related_dtc, relevance_score=0.85)
                        )
                        existing_codes.add(related_node.code)
    except Exception as e:
        logger.warning(f"Error fetching Neo4j related codes: {sanitize_log(str(e))}")

    # If we still need more, find codes in the same category with similar prefix
    if len(results) < limit:
        # Same category codes (P0xxx for P0101, etc.)
        prefix = code[:2]
        stmt_results = await repository.search(prefix, limit=limit + len(existing_codes))

        for r in stmt_results:
            if r.code not in existing_codes:
                results.append(_dtc_model_to_search_result(r, relevance_score=0.5))
                existing_codes.add(r.code)

                if len(results) >= limit:
                    break

    # Sort by relevance
    results.sort(key=lambda x: x.relevance_score or 0, reverse=True)

    return results[:limit]


@router.post(
    "/",
    response_model=DTCCode,
    status_code=status.HTTP_201_CREATED,
    responses=CREATE_DTC_RESPONSES,
    summary="Create DTC code",
    description="""
**Create a new DTC code entry** in the database.

Required fields:
- `code`: DTC code (e.g., P0101)
- `description_en`: English description
- `category`: One of powertrain, body, chassis, network
- `severity`: One of low, medium, high, critical

Optional fields:
- `description_hu`: Hungarian translation
- `symptoms`: List of symptoms
- `possible_causes`: List of causes
- `diagnostic_steps`: List of repair steps
- `related_codes`: List of related DTC codes
    """,
)
async def create_dtc_code(
    dtc_data: DTCCreate,
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new DTC code entry.

    Args:
        dtc_data: DTC code data
        db: Database session

    Returns:
        Created DTC code

    Raises:
        400: DTC code already exists
    """
    repository = DTCCodeRepository(db)

    # Check if code already exists
    existing = await repository.get_by_code(dtc_data.code.upper())
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"DTC code {dtc_data.code} already exists",
        )

    # Create the DTC code
    dtc = await repository.create(
        {
            "code": dtc_data.code.upper(),
            "description_en": dtc_data.description_en,
            "description_hu": dtc_data.description_hu,
            "category": dtc_data.category.value,
            "severity": dtc_data.severity.value,
            "is_generic": dtc_data.is_generic,
            "system": dtc_data.system,
            "symptoms": dtc_data.symptoms,
            "possible_causes": dtc_data.possible_causes,
            "diagnostic_steps": dtc_data.diagnostic_steps,
            "related_codes": dtc_data.related_codes,
        }
    )

    await db.commit()

    logger.info(f"Created DTC code: {sanitize_log(dtc.code)}")

    return DTCCode(
        code=dtc.code,
        description_en=dtc.description_en,
        description_hu=dtc.description_hu,
        category=dtc.category,
        is_generic=dtc.is_generic,
    )


@router.post(
    "/bulk",
    response_model=Dict[str, Any],
    status_code=status.HTTP_201_CREATED,
    responses=BULK_IMPORT_RESPONSES,
    summary="Bulk import DTC codes",
    description="""
**Bulk import multiple DTC codes** at once.

Request body:
- `codes`: Array of DTC code objects
- `overwrite_existing`: If true, update existing codes; if false, skip them

Response includes:
- `created`: Number of new codes created
- `updated`: Number of existing codes updated
- `skipped`: Number of codes skipped (already exist, overwrite=false)
- `errors`: List of codes that failed to import
- `total`: Total number of codes in request
    """,
)
async def bulk_import_dtc_codes(
    import_data: DTCBulkImport,
    db: AsyncSession = Depends(get_db),
):
    """
    Bulk import DTC codes.

    Args:
        import_data: Bulk import data with list of codes
        db: Database session

    Returns:
        Import summary with created and skipped counts
    """
    repository = DTCCodeRepository(db)

    created = 0
    updated = 0
    skipped = 0
    errors = []

    for dtc_data in import_data.codes:
        try:
            code = dtc_data.code.upper()
            existing = await repository.get_by_code(code)

            data = {
                "code": code,
                "description_en": dtc_data.description_en,
                "description_hu": dtc_data.description_hu,
                "category": dtc_data.category.value,
                "severity": dtc_data.severity.value,
                "is_generic": dtc_data.is_generic,
                "system": dtc_data.system,
                "symptoms": dtc_data.symptoms,
                "possible_causes": dtc_data.possible_causes,
                "diagnostic_steps": dtc_data.diagnostic_steps,
                "related_codes": dtc_data.related_codes,
            }

            if existing:
                if import_data.overwrite_existing:
                    await repository.update(existing.id, data)
                    updated += 1
                else:
                    skipped += 1
            else:
                await repository.create(data)
                created += 1

        except Exception as e:
            errors.append({"code": dtc_data.code, "error": str(e)})
            logger.error(
                f"Error importing DTC {sanitize_log(dtc_data.code)}: {sanitize_log(str(e))}"
            )

    await db.commit()

    logger.info(f"Bulk import complete: {created} created, {updated} updated, {skipped} skipped")

    return {
        "created": created,
        "updated": updated,
        "skipped": skipped,
        "errors": errors,
        "total": len(import_data.codes),
    }
