"""
Service shop search endpoints for Hungarian auto repair shops.

Provides endpoints to:
- Search service shops by region, vehicle make, service type
- List available regions
- Get details for a specific shop
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.v1.endpoints.auth import get_optional_current_user
from app.api.v1.schemas.services import (
    Region,
    ServiceSearchResponse,
    ServiceShop,
)
from app.core.logging import get_logger
from app.services.service_shop_service import get_service_shop_service

router = APIRouter()
logger = get_logger(__name__)


# =============================================================
# Search service shops
# =============================================================


@router.get(
    "/search",
    response_model=ServiceSearchResponse,
    summary="Szerviz keresés",
    description="""
**Autószervizek keresése** régió, márka és szolgáltatás típus alapján.

Támogatott szűrők:
- **region**: Régió szűrő (pl. "Budapest", "Pest megye")
- **vehicle_make**: Jármű márka szűrő (pl. "Volkswagen")
- **service_type**: Szolgáltatás típus (pl. "olajcsere", "fékjavítás")
- **sort_by**: Rendezés (rating, distance, name)
- **lat/lng**: Koordináták a távolság alapú rendezéshez

**Példák:**
- `/search?region=Budapest` - Budapesti szervizek
- `/search?vehicle_make=Toyota&sort_by=rating` - Toyota szervizek
- `/search?lat=47.497&lng=19.040&sort_by=distance` - Legközelebbi
    """,
)
async def search_service_shops(
    region: Optional[str] = Query(None, description="Régió szűrő (pl. Budapest)"),
    vehicle_make: Optional[str] = Query(None, description="Jármű márka szűrő"),
    service_type: Optional[str] = Query(None, description="Szolgáltatás típus szűrő"),
    sort_by: Optional[str] = Query("rating", description="Rendezés: rating, distance, name"),
    lat: Optional[float] = Query(None, description="Szélességi fok (távolság rendezéshez)"),
    lng: Optional[float] = Query(None, description="Hosszúsági fok (távolság rendezéshez)"),
    limit: int = Query(20, ge=1, le=100, description="Találatok maximális száma"),
    offset: int = Query(0, ge=0, description="Kihagyandó találatok száma"),
    _current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
) -> ServiceSearchResponse:
    """
    Autószervizek keresése különböző szűrők alapján.

    Args:
        region: Régió szűrő
        vehicle_make: Jármű márka szűrő
        service_type: Szolgáltatás típus szűrő
        sort_by: Rendezés módja
        lat: Szélességi fok
        lng: Hosszúsági fok
        limit: Találatok maximális száma
        offset: Kihagyandó találatok száma

    Returns:
        Szervizek listája és összesített darabszám
    """
    try:
        service = get_service_shop_service()
        result = service.search_shops(
            region=region,
            vehicle_make=vehicle_make,
            service_type=service_type,
            sort_by=sort_by,
            lat=lat,
            lng=lng,
            limit=limit,
            offset=offset,
        )

        shops = [ServiceShop(**shop) for shop in result["shops"]]
        total = result["total"]

        return ServiceSearchResponse(
            shops=shops,
            total=total,
            limit=limit,
            offset=offset,
            has_more=offset + len(shops) < total,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Szerviz keresési hiba: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Nem sikerült a szervizek keresése. Kérjük, próbálja újra később.",
        )


# =============================================================
# List available regions
# =============================================================


@router.get(
    "/regions",
    response_model=List[Region],
    summary="Elérhető régiók",
    description="""
**Elérhető régiók listázása** a szerviz kereséshez.

Visszaadja az összes régiót, amelyekben szerviz található.
    """,
)
async def get_regions(
    _current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
) -> List[Region]:
    """
    Elérhető régiók listázása.

    Returns:
        Régiók listája
    """
    try:
        service = get_service_shop_service()
        regions_data = service.get_regions()

        return [Region(**region) for region in regions_data]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Régiók lekérdezési hiba: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Nem sikerült a régiók lekérdezése. Kérjük, próbálja újra később.",
        )


# =============================================================
# Get single shop by ID
# =============================================================


@router.get(
    "/{shop_id}",
    response_model=ServiceShop,
    summary="Szerviz részletei",
    description="""
**Egyetlen szerviz részletes adatai** azonosító alapján.

Visszaadja a szerviz teljes adatlapját: cím, nyitvatartás,
szolgáltatások, értékelések.
    """,
)
async def get_shop_by_id(
    shop_id: str,
    _current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
) -> ServiceShop:
    """
    Szerviz lekérdezése azonosító alapján.

    Args:
        shop_id: A szerviz egyedi azonosítója

    Returns:
        Szerviz részletes adatai

    Raises:
        404: A szerviz nem található
    """
    try:
        service = get_service_shop_service()
        shop = service.get_shop_by_id(shop_id)

        if shop is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="A keresett szerviz nem található.",
            )

        return ServiceShop(**shop)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Szerviz lekérdezési hiba ({shop_id}): {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Nem sikerült a szerviz lekérdezése. Kérjük, próbálja újra később.",
        )
